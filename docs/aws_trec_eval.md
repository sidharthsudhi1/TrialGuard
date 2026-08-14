# TREC 2021 agent eval on EC2 + S3

Written 2026-08-13. This is an ops plan, not a product feature. SageMaker is
deliberately out of scope here; see "Why not SageMaker" below.

## Why this plan exists

Three facts collided:

1. **v4 (inclusion + exclusion) is untested on a cohort that has exclusion
   criteria.** The SIGIR A/B (`data/reports/phase9v4_agent_sigir.json`) finished
   cleanly, but SIGIR's parser yields **zero** exclusion criteria across 2,991
   trials. The headline movements on that run (trial accuracy 0.2611 → 0.3722,
   unsupported rate 0.0287 → 0.0966) are from raising `MAX_CRITERIA` 12 → 24,
   not from exclusion handling. TREC 2021 is the cohort that can actually
   exercise v4: 23,930 / 26,149 trials have exclusion text.
2. **The Mac cannot load TREC.** An 8 GB machine already at 6.5 GB used crashed
   while probing `get_index("trec_2021")`. That call builds in-memory BM25 over
   all 26k trials, then `_build_subset` loads and normalises the corpus a second
   time. SIGIR (3k trials) peaked at 366 MB. TREC is ~9× larger. This is not a
   "close some tabs" problem.
3. **The agent eval does not need that index.** `_build_subset` only uses
   `get_index()` for `corpus_ids` and fallback `trial_texts`. Both are available
   from the jsonl + qrels. The BM25/numpy index is a retrieval artifact. Loading
   it for an analyst A/B is why the Mac died.

So: run the TREC A/B on a short-lived CPU box, persist the gitignored corpus and
the report in S3, terminate the box. Inference stays on DeepInfra (~$0.06). The
box is RAM + disk + network, nothing else.

This is also the right first AWS project: one bucket, one instance, one secret,
then shut it down. It teaches IAM, S3, EC2, and Parameter Store without pretending
a 30-minute HTTP loop is a training job.

## Why EC2 + S3 (and not the alternatives)

| Option | Verdict |
|---|---|
| This Mac | Already crashed. 1.5 GB free cannot hold TREC FileIndex. |
| GitHub Actions | ~7 GB RAM, 6 h cap, TREC jsonl is gitignored. Same OOM risk, worse to debug. |
| Hugging Face Spaces | Public demo. Do not put `DEEPINFRA_API_KEY` there. |
| Lambda | 15-minute timeout. SIGIR took 29 minutes. |
| SageMaker Studio | Correct product to learn later (MedCPT batch transform / endpoint). Wrong product for this script. Studio notebooks do not stop when a cell finishes; that is how the bill happens. |
| `t3.micro` free tier | 1 GB RAM. Smaller than the Mac. Will die the same way. |
| **EC2 `t3.medium` + S3** | 4 GB is enough **after** the FileIndex skip. S3 holds the 131 MB corpus and the report so terminating the instance does not lose the result. Cost: cents. |

Region: `ap-southeast-2` (Sydney). Matches the machine in UTC+10; lower SSH
latency; data does not leave the AU region.

Expected spend for one run:

- EC2 `t3.medium`: ~$0.04 / hour, budget 2 hours → ~$0.08
- S3: 131 MB + report + cache, pennies
- DeepInfra: ~$0.06 (same order as SIGIR v4)
- **Total: under $0.20 if the instance is terminated the same day**

The expensive outcome is leaving the instance running. The last step of this
plan is terminate, then confirm in the console that instance state is
`terminated` and there is no Studio / SageMaker app.

## Success criteria

The run is done when all of these are true:

- `data/reports/phase9v4_agent_trec_2021.json` exists locally (pulled from S3)
- `verified.by_kind.exclusion.n_criteria` **> 0** (the SIGIR run was 0; this is
  the actual v4 test)
- `n_trials` is in the same ballpark as SIGIR (180 if `--n-patients 30 --per-class 3`)
- `significance.fisher_p` is present (A/B completed both arms)
- EC2 instance is **terminated**, not stopped
- DeepInfra spend for the day is still under `DAILY_USD_CAP` (default $2)

## Prerequisite: stop loading FileIndex in the agent eval

Do this **before** launching EC2. Without it, use `t3.large` (8 GB) and still
risk a slow BM25 build.

Change `_build_subset` in `src/trialguard/eval/agent_metrics.py` so TREC/SIGIR
agent eval:

1. Loads patients + qrels via `cohorts.py` (already committed).
2. Loads the corpus jsonl via `_load_trec_trials` / `_load_sigir_trials`.
3. Normalises only what it needs (or streams and keeps only chosen NCT IDs).
4. Treats "in corpus" as "present in the jsonl", not "present in FileIndex".
5. Uses `eligibility_raw` as `source_text`. Never calls `get_index()`.

Keep `get_index()` for retrieval eval (`eval_retrieval.py`). Only the analyst
A/B should skip it.

Sanity check on the Mac after the patch (will not OOM):

```bash
.venv/bin/python -c "
from trialguard.eval.agent_metrics import _build_subset
sub = _build_subset('trec_2021', 2, 1)
n = sum(len(t['criteria']) for p in sub for t in p['trials'])
ne = sum(1 for p in sub for t in p['trials'] for c in t['criteria'] if c['kind']=='exclusion')
print(f'trials={sum(len(p[\"trials\"]) for p in sub)} criteria={n} exclusion={ne}')
assert ne > 0, 'TREC subset must contain exclusion criteria'
"
```

Commit and push that patch to the branch EC2 will clone. The instance must not
run current `main` if `main` still calls `get_index()`.

---

## Concrete plan

Do the AWS console work yourself. That is the learning. Commands below are the
exact CLI; the console equivalents are named in each step.

### 0. Account and local CLI

1. Create / log in to an AWS account. Enable MFA on the root user. Do not use
   the root user for the rest of this plan.
2. Create an IAM user `trialguard-ops` with console + access keys, attach
   `AdministratorAccess` **only if this is a throwaway learning account**. On a
   real account, use the tighter policy in the appendix.
3. Install AWS CLI v2 on the Mac. Configure:

```bash
aws configure
# AWS Access Key ID:     <trialguard-ops>
# AWS Secret Access Key: <secret>
# Default region:        ap-southeast-2
# Default output:        json
```

4. Confirm:

```bash
aws sts get-caller-identity
aws ec2 describe-availability-zones --query 'AvailabilityZones[].ZoneName'
```

### 1. S3 bucket

Name must be globally unique. Pick one and keep it:

```text
trialguard-eval-<your-initials>-apse2
```

Console: S3 → Create bucket.

- Region: `ap-southeast-2`
- Block **all** public access: on
- Versioning: off
- Default encryption: SSE-S3
- Object Ownership: Bucket owner enforced

CLI equivalent:

```bash
BUCKET=trialguard-eval-<your-initials>-apse2

aws s3api create-bucket \
  --bucket "$BUCKET" \
  --create-bucket-configuration LocationConstraint=ap-southeast-2 \
  --region ap-southeast-2

aws s3api put-public-access-block --bucket "$BUCKET" \
  --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
```

Layout we will use:

```text
s3://$BUCKET/eval/trec_2021/trec_2021_corpus.jsonl   # optional cache of NCBI file
s3://$BUCKET/reports/phase9v4_agent_trec_2021.json   # the deliverable
s3://$BUCKET/cache/analyst/                          # so a retry costs $0
```

Optional, from the Mac, if the local 131 MB jsonl survived the crash:

```bash
aws s3 cp data/eval/trec_2021/trec_2021_corpus.jsonl \
  "s3://$BUCKET/eval/trec_2021/trec_2021_corpus.jsonl"
```

If it did not, skip this. The instance downloads from NCBI FTP
(`trialguard.eval.corpus_loader.TREC_SOURCES`) and can upload to S3 after.

### 2. Secret in SSM Parameter Store

Console: Systems Manager → Parameter Store → Create parameter.

- Name: `/trialguard/deepinfra_api_key`
- Type: **SecureString**
- Value: the DeepInfra key from local `.env` (`DEEPINFRA_API_KEY=`)
- KMS: `alias/aws/ssm`

Do not put the key in EC2 user-data, in an S3 object, or in a GitHub secret for
this run. User-data is visible on the instance and in CloudTrail.

```bash
# paste the key when prompted; it will not echo if you read from the tty
aws ssm put-parameter \
  --name /trialguard/deepinfra_api_key \
  --type SecureString \
  --value "$(grep '^DEEPINFRA_API_KEY=' .env | cut -d= -f2-)"
```

Confirm the name exists without printing the value:

```bash
aws ssm get-parameter --name /trialguard/deepinfra_api_key --query 'Parameter.Name'
```

### 3. IAM role for the instance

The box should not have your access keys on disk. It assumes a role.

Console: IAM → Roles → Create role → AWS service → EC2.

Name: `trialguard-trec-ec2`

Trust: EC2. Permissions (inline policy `trialguard-trec-s3-ssm`):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3EvalArtifacts",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::trialguard-eval-<your-initials>-apse2",
        "arn:aws:s3:::trialguard-eval-<your-initials>-apse2/*"
      ]
    },
    {
      "Sid": "ReadDeepInfraKey",
      "Effect": "Allow",
      "Action": ["ssm:GetParameter"],
      "Resource": "arn:aws:ssm:ap-southeast-2:*:parameter/trialguard/deepinfra_api_key"
    },
    {
      "Sid": "DecryptSecureString",
      "Effect": "Allow",
      "Action": ["kms:Decrypt"],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "kms:ViaService": "ssm.ap-southeast-2.amazonaws.com"
        }
      }
    }
  ]
}
```

Replace the bucket name. No `s3:*`, no `ssm:*`.

### 4. Security group and key pair

Key pair (Console: EC2 → Key pairs → Create):

- Name: `trialguard-trec`
- Type: ED25519 (or RSA if the Mac's ssh client is old)
- Download `trialguard-trec.pem`. It will not be offered again.
- `chmod 400 trialguard-trec.pem`
- The pem is gitignored (`*.pem`). Keep it out of the repo.

Security group `trialguard-trec-ssh`:

- Inbound: TCP 22 from **your public IP only** (`x.x.x.x/32`). Not `0.0.0.0/0`.
- Outbound: all (instance needs NCBI FTP, GitHub, DeepInfra, S3, SSM).

```bash
MYIP=$(curl -s https://checkip.amazonaws.com)

aws ec2 create-security-group \
  --group-name trialguard-trec-ssh \
  --description "SSH from my IP only, TREC eval"

aws ec2 authorize-security-group-ingress \
  --group-name trialguard-trec-ssh \
  --protocol tcp --port 22 \
  --cidr "${MYIP}/32"
```

### 5. Launch EC2

Console: EC2 → Launch instance.

| Field | Value |
|---|---|
| Name | `trialguard-trec-eval` |
| AMI | Ubuntu Server 24.04 LTS (x86_64) |
| Instance type | **`t3.medium`** (2 vCPU, 4 GB). Not micro. Not GPU. |
| Key pair | `trialguard-trec` |
| Firewall | existing group `trialguard-trec-ssh` |
| Storage | 20 GB gp3 (AMI default 8 GB is tight once venv + corpus land) |
| IAM instance profile | `trialguard-trec-ec2` |
| Auto-assign public IP | enable |

Do not enable Hibernation. Do not request a GPU. Do not add a second NIC.

CLI equivalent (fill subnet / sg / ami from the console picker the first time;
AMI IDs change):

```bash
# After first console launch you can ignore this block.
# Prefer the console for the first run so you see every checkbox.
```

Wait until Instance state is `running` and Status checks are `2/2`. Copy the
public IPv4.

### 6. SSH in and install the runtime

```bash
chmod 400 trialguard-trec.pem
ssh -i trialguard-trec.pem ubuntu@<PUBLIC_IP>
```

On the instance:

```bash
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev git unzip
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
unzip -q awscliv2.zip && sudo ./aws/install && rm -rf aws awscliv2.zip

# instance profile should already give credentials
aws sts get-caller-identity
```

If `get-caller-identity` fails, the instance profile is not attached. Fix that
in the console (Actions → Security → Modify IAM role) before continuing. Do not
`aws configure` access keys onto the box.

Clone the branch that contains the FileIndex skip:

```bash
git clone -b <branch-with-skip> https://github.com/<you>/TrialGuard.git
cd TrialGuard
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

`sentence-transformers` / torch will install. That is expected (package
dependency). We will not load MedCPT if FileIndex is skipped. If RAM looks
tight during `pip install`, that is fine; the crash was BM25 + 26k dicts, not
pip.

### 7. Corpus, secret, env

```bash
BUCKET=trialguard-eval-<your-initials>-apse2
export AWS_DEFAULT_REGION=ap-southeast-2

# Prefer S3 if you uploaded from the Mac; else NCBI FTP
if aws s3 ls "s3://$BUCKET/eval/trec_2021/trec_2021_corpus.jsonl"; then
  mkdir -p data/eval/trec_2021
  aws s3 cp "s3://$BUCKET/eval/trec_2021/trec_2021_corpus.jsonl" \
    data/eval/trec_2021/trec_2021_corpus.jsonl
else
  python -m trialguard.scripts.load_eval_corpus
  aws s3 cp data/eval/trec_2021/trec_2021_corpus.jsonl \
    "s3://$BUCKET/eval/trec_2021/trec_2021_corpus.jsonl"
fi

# qrels + queries are already in git (data/eval/trec_2021/*.tsv, *.jsonl)

export DEEPINFRA_API_KEY="$(aws ssm get-parameter \
  --name /trialguard/deepinfra_api_key --with-decryption \
  --query 'Parameter.Value' --output text)"
export LLM_PROVIDER=deepinfra
export TG_PROMPT_VERSION=v4
export TG_ANALYST_DELAY=0
export TG_EVAL_WORKERS=4
export TRACING_ENABLED=false
export DAILY_USD_CAP=2.00
```

Workers = 4, not 6: 4 GB box, threads share RAM, DeepInfra is the bottleneck.
`TRACING_ENABLED=false` avoids Langfuse timeouts (they spammed the SIGIR log
and are unrelated to the metric).

Sanity check **before** the paid loop:

```bash
python -c "
from trialguard.eval.agent_metrics import _build_subset
sub = _build_subset('trec_2021', 2, 1)
ne = sum(1 for p in sub for t in p['trials'] for c in t['criteria'] if c['kind']=='exclusion')
print('exclusion criteria in tiny subset:', ne)
assert ne > 0
"
```

If this asserts, stop. Do not start the A/B. The skip patch is missing or the
jsonl did not parse.

### 8. Run the A/B inside tmux

SSH drops must not kill the job.

```bash
sudo apt-get install -y tmux
tmux new -s trec
```

```bash
source .venv/bin/activate
# re-export env vars inside tmux (they do not survive a new shell)

python -u -m trialguard.eval.agent_metrics \
  --cohort trec_2021 \
  --n-patients 30 \
  --per-class 3 \
  --tag phase9v4 \
  2>&1 | tee data/reports/phase9v4_trec_run.log
```

Detach: `Ctrl-b` then `d`. Reattach: `tmux attach -t trec`.

Wall clock: ~30–60 minutes (SIGIR was 29 min at 6 workers). Watch RAM:

```bash
# other SSH window
free -h
```

If `available` drops under ~300 MB, lower `TG_EVAL_WORKERS` to 2 and restart.
Cache hits resume; only in-flight calls are re-paid.

Every few minutes, snapshot cache + partial log to S3 so a crash is not a
total loss:

```bash
aws s3 sync data/cache/analyst "s3://$BUCKET/cache/analyst"
aws s3 cp data/reports/phase9v4_trec_run.log "s3://$BUCKET/reports/"
```

### 9. Pull the report, then terminate

When the JSON prints and the process exits 0:

```bash
python -c "
import json
r=json.load(open('data/reports/phase9v4_agent_trec_2021.json'))
print('n_trials', r['verified']['n_trials'])
print('trial_accuracy', r['verified']['trial_accuracy'])
print('unsupported', r['verified']['unsupported_verdict_rate'])
print('exclusion n', r['verified']['by_kind']['exclusion']['n_criteria'])
print('exclusion unsupported', r['verified']['by_kind']['exclusion']['unsupported_verdict_rate'])
print('fisher_p', r['significance']['fisher_p'])
assert r['verified']['by_kind']['exclusion']['n_criteria'] > 0
"

aws s3 cp data/reports/phase9v4_agent_trec_2021.json \
  "s3://$BUCKET/reports/phase9v4_agent_trec_2021.json"
aws s3 sync data/cache/analyst "s3://$BUCKET/cache/analyst"
```

On the Mac:

```bash
aws s3 cp "s3://$BUCKET/reports/phase9v4_agent_trec_2021.json" \
  data/reports/phase9v4_agent_trec_2021.json
```

Then **terminate** (not stop):

```bash
# from the Mac, using the instance id from the console
aws ec2 terminate-instances --instance-ids i-xxxxxxxx
aws ec2 wait instance-terminated --instance-ids i-xxxxxxxx
aws ec2 describe-instances --instance-ids i-xxxxxxxx \
  --query 'Reservations[].Instances[].State.Name'
```

Confirm in the console: Instances → `terminated`. There should be no SageMaker
domain, no EBS volume left in `available` (the root volume dies with the
instance unless you unchecked Delete on termination — leave that checked).

Keep the S3 bucket. The report and analyst cache are the reason the instance
was allowed to die.

### 10. What to do with the numbers

Read `verified.by_kind` first, not the aggregate.

- If exclusion unsupported-rate ≫ inclusion: v4 quoting of exclusion text is
  the bug. Next step is a v5 prompt, not a baseline re-anchor.
- If both kinds are similarly worse than Phase 8 SIGIR: the `MAX_CRITERIA`
  12 → 24 change is still in play; isolate with a cap=12 rerun (cache-only,
  $0) before touching prompts.
- Do **not** point `data/reports/baselines.json` at this file until the
  exclusion vs cap story is explicit. The gate is still anchored to
  `phase8di_agent_sigir.json` on purpose.

## What this plan does not do

- Does not deploy the Gradio demo.
- Does not put MedCPT on SageMaker. That is the follow-up learning project:
  batch-transform the 26k corpus, optionally host an endpoint. Separate doc.
- Does not open the security group to the world.
- Does not store API keys in git, user-data, or S3.
- Does not leave a GPU or Studio app running.

## Failure table

| Symptom | Likely cause | Action |
|---|---|---|
| Mac / instance OOM during `_build_subset` | `get_index()` still called | Abort. Confirm the skip patch is on the cloned branch. |
| `exclusion n_criteria == 0` | Wrong corpus or SIGIR command | You ran `--cohort sigir`. TREC jsonl must be the 26k file. |
| `AccessDenied` on S3 or SSM | Instance profile missing or wrong bucket ARN | Attach `trialguard-trec-ec2`; do not paste access keys. |
| SSH timeout | Wrong SG CIDR (home IP changed) | Update inbound 22 to the new `/32`. |
| DeepInfra 401 | SSM parameter empty / `--with-decryption` forgotten | `get-parameter --with-decryption`; never print it into the log. |
| Eval dies at 20 min, empty report | SSH session died, no tmux | Restart inside tmux; cache resumes. |
| Bill surprise | Instance left `running` or `stopped` | `stopped` still pays for EBS. **Terminate.** |

## Appendix: tighter IAM for the laptop user

If the AWS account is not throwaway, replace `AdministratorAccess` on
`trialguard-ops` with: `AmazonEC2FullAccess` scoped later, `AmazonS3FullAccess`
scoped to this bucket, `AmazonSSMFullAccess` scoped to `/trialguard/*`, plus
`iam:PassRole` on `trialguard-trec-ec2` only. For a first learning pass, a
fresh account with MFA and a calendar reminder to terminate is enough.

## Appendix: teardown

When the report is committed and you do not need a retry:

```bash
aws s3 rm "s3://$BUCKET" --recursive
aws s3api delete-bucket --bucket "$BUCKET"
aws ssm delete-parameter --name /trialguard/deepinfra_api_key
aws ec2 delete-security-group --group-name trialguard-trec-ssh
aws ec2 delete-key-pair --key-name trialguard-trec
rm -f trialguard-trec.pem
# IAM role + instance profile: delete in console after the instance is terminated
```

Do not delete the local `data/reports/phase9v4_agent_trec_2021.json`. That is
the artifact this whole plan exists to produce.
