# DevOps plan: CI/CD, GitOps, workflows, observability, EKS

Branch: `devops/ci-cd`, cut from `main` at `3bccde2`. Design source: `docs/interview_qa.md` Q25–Q28, `distributed_plan.md`
Phase 4–5. Rule from Q27: **each resource has exactly one owner.** Terraform owns cloud
resources, Argo CD owns in-cluster app state, Airflow owns scheduled workflows.

Ordered so every phase is usable on its own and the expensive cloud step comes last.

| # | Phase | Delivers | Done when |
|---|---|---|---|
| 1 | **Replay eval + regression gate in CI** | `scripts/eval_gate.py` compares a replayed sweep to a committed baseline: per-archetype floors, per-question flips (paired), golden must-pass set, compared only on the intersection of question IDs. CI restores a database dump (corpus + journal) so it can replay at $0. New CI job `eval-gate`. | A PR that makes a previously-correct question wrong fails CI; replay costs $0.00 |
| 2 | **Image build + publish (CI → GHCR)** | Workflow builds the Dockerfile on every PR, pushes `ghcr.io/<owner>/financevault:<sha>` on `main`, then a GitOps commit bumps the image tag in `infra/k8s/overlays/*` | Merge to main → new tag in git, no manual `kubectl` |
| 3 | **Observability in the app** | `/metrics` (Prometheus: requests, run outcomes, model calls live vs cached, $ spent, run duration). OpenTelemetry traces: HTTP request → run → model call → tool call, `run_id` attribute. Local stack in compose: OTel Collector, Prometheus, Tempo, Grafana with a provisioned dashboard in `monitoring/dashboards/`. | A slow run can be traced to the tool call that took the time, without logs |
| 4 | **Kubernetes manifests (Kustomize)** | `infra/k8s/base`: API Deployment (probes on `/healthz` `/readyz`, `terminationGracePeriodSeconds` > 300 s run bound), Service, PDB, migration Job as Argo CD PreSync hook. Overlays `local` (kind) and `aws`. | `kubectl kustomize` renders; app runs on kind |
| 5 | **Argo CD (GitOps)** | App-of-apps in `infra/argocd`: the FinanceVault app plus kube-prometheus-stack and the OTel Collector as Helm apps, `automated: {prune, selfHeal}`. Bootstrap script for kind. | Manual `kubectl edit` gets reverted; git is the only way to change the cluster |
| 6 | **Airflow** | `airflow/dags`: nightly ingest (per company, `sec_api` pool keeps total SEC traffic ≤ 10 req/s, no retry on 403) and nightly eval (replay sweep + gate, live sweep on demand). DAG import test in CI. | DAGs parse in CI; ingest task retries are safe (upserts) |
| 7 | **Terraform → AWS EKS** (last) | `infra/terraform`: VPC, EKS (managed CPU node group, GPU group off by default), RDS Postgres 17 + pgvector, ECR, IRSA roles, Argo CD bootstrapped via Helm. `terraform fmt/validate` in CI. **No `apply` without your go-ahead — it costs money.** | `terraform plan` is clean; one `apply` + Argo CD sync gives a running stack |

## Decisions and constraints

- **The eval gate replays, it doesn't call models.** A live sweep costs about $2 (Q25), so it runs
  nightly in Airflow or by hand, never on every PR.
- **CI fixture is a database dump, not a re-ingest.** Tool output includes chunk ids (serial
  keys), so a fresh ingest can renumber them, change every prompt, and turn the replay into
  journal misses. CI restores `make dump` output from the `FV_CI_FIXTURE_URL` secret. The repo is
  public, so where the dump lives (it holds the journal) is your call.
- **Single API process.** Run streams live in process memory (`api/streams.py`), so the
  Deployment runs `replicas: 1` until the job queue from `distributed_plan.md` exists. HPA is out
  of scope for now.
- **App code this branch carries.** The image needs the HTTP API (`api/app.py`, the demo page,
  `Dockerfile`), which was never on `main`; it comes in here with its tests (`tests/test_api.py`).
  `main` is also getting BM25 retrieval from another branch: once that lands, merge it here and
  rebuild the gate's baseline with `make catalog`, since changed search results change every prompt.

## Status

| Phase | State | Evidence |
|---|---|---|
| 1 | **Built**, CI job waits on `FV_CI_FIXTURE_URL` | `make gate`: 71/71 question-policy pairs reproduce, 3.6 s, $0.00. A one-space change to `executor.SYSTEM` gives 4/4 journal-miss regressions on `--limit 3`, exit 1. A tampered baseline gives drift, exit 1. `tests/test_eval_gate.py`: 7 pass. |
| 2 | **Built (build + push)**; GitOps tag bump lands with Phase 4 | `.github/workflows/image.yml`, owner name lowercased for GHCR. Not yet run on GitHub. |
| 3–7 | Not started | |
