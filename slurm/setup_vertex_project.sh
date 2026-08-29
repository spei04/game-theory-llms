#!/usr/bin/env bash
#
# Point the cluster's Gemini calls at Vertex AI in an existing GCP project, so
# the spend bills that project's billing account.
#
# Default target is your-gcp-project, which is already set up for this:
#   - aiplatform.googleapis.com enabled
#   - billing enabled on XXXXXX-XXXXXX-XXXXXX (the org-owned credits)
#   - iam.managed.disableServiceAccountKeyCreation overridden to enforce:false
#     at the project level, so SA keys are allowed here even though the
#     your-org.example.com org blocks them everywhere else
#
# A service-account key is used rather than your own login because the domain's
# reauth policy expires user credentials within the hour. ADC user creds would
# die partway through a 12-hour SLURM array; a service account does not reauth.
#
# Run LOCALLY (needs browser-authenticated gcloud), then copy the key to the
# cluster and add the env block to $REPO/.env there.
#
# Usage:
#   ./slurm/setup_vertex_project.sh --dry-run
#   ./slurm/setup_vertex_project.sh
#   ./slurm/setup_vertex_project.sh --project your-gcp-project --budget 100
#
# Flags:
#   --project <id>       project to bill (default: your-gcp-project)
#   --location <region>  Vertex location (default: global)
#   --key-dir <path>     where to write the SA key (default: ~/.gcp)
#   --cluster <host>     cluster host for the printed scp command
#   --budget <usd>       create a budget alert scoped to Vertex spend on this
#                        project; omit to skip budget creation
#   --sa-name <name>     service account name (default: gemini-api)
#   --dry-run            print every command instead of running it
#
# ON CREDITS VS YOUR CARD
# -----------------------
# Credits attach to the BILLING ACCOUNT, not the project. XXXXXX-XXXXXX-XXXXXX
# is a paid account (open, seven production projects), not a Free Trial account.
# A Free Trial account halts when credits run out; a paid one silently rolls
# charges onto the card. Credits are a discount, not a cap.
#
# Real protection, strongest first:
#   1. COST_CAP in the simulation (--cost_cap on run_baseline_array.sh, default
#      $50/task). Aborts the run. The only per-run hard stop.
#   2. Budget alerts (--budget). Notify only; they stop nothing.
#   3. A true kill switch: budget -> Pub/Sub -> Cloud Function calling
#      projects.updateBillingInfo to unlink billing. Not set up here.
#      https://cloud.google.com/billing/docs/how-to/disable-billing-with-notifications
#
# Verify your credits actually cover Vertex AI before a big run — gcloud cannot
# list credits, so check the console:
#   https://console.cloud.google.com/billing/XXXXXX-XXXXXX-XXXXXX/credits

set -euo pipefail

PROJECT_ID="your-gcp-project"
LOCATION="global"
KEY_DIR="$HOME/.gcp"
CLUSTER="USERNAME@<cluster>"
BUDGET_AMOUNT=""
DRY_RUN=""
SA_NAME="gemini-api"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project)   PROJECT_ID="$2"; shift 2 ;;
        --location)  LOCATION="$2"; shift 2 ;;
        --key-dir)   KEY_DIR="$2"; shift 2 ;;
        --cluster)   CLUSTER="$2"; shift 2 ;;
        --budget)    BUDGET_AMOUNT="$2"; shift 2 ;;
        --sa-name)   SA_NAME="$2"; shift 2 ;;
        --dry-run)   DRY_RUN="1"; shift 1 ;;
        -h|--help)   sed -n '2,50p' "$0"; exit 0 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

run() {
    if [ -n "$DRY_RUN" ]; then
        echo "  [dry-run] $*"
    else
        "$@"
    fi
}

step() { echo; echo "==> $*"; }

SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
KEY_PATH="${KEY_DIR}/${PROJECT_ID}-${SA_NAME}.json"

# ---------------------------------------------------------------------------
# Preflight — verify every assumption before creating anything
# ---------------------------------------------------------------------------
step "Preflight"
command -v gcloud >/dev/null || { echo "gcloud not found." >&2; exit 1; }
if ! gcloud projects list --limit=1 >/dev/null 2>&1; then
    echo "ERROR: gcloud cannot refresh credentials. Run 'gcloud auth login' first." >&2
    exit 1
fi
echo "    authenticated as $(gcloud config get-value account 2>/dev/null)"

gcloud projects describe "$PROJECT_ID" >/dev/null 2>&1 \
    || { echo "ERROR: project $PROJECT_ID not found or not visible." >&2; exit 1; }

BILLING_ACCOUNT=$(gcloud billing projects describe "$PROJECT_ID" \
    --format="value(billingAccountName)" 2>/dev/null | sed 's|billingAccounts/||')
BILLING_ENABLED=$(gcloud billing projects describe "$PROJECT_ID" \
    --format="value(billingEnabled)" 2>/dev/null)
if [ "$BILLING_ENABLED" != "True" ]; then
    echo "ERROR: billing is not enabled on $PROJECT_ID. Vertex calls will fail." >&2
    exit 1
fi
echo "    billing account: $BILLING_ACCOUNT (enabled)"

# SA key creation is blocked org-wide; this project needs an explicit override.
KEY_POLICY=$(gcloud org-policies describe iam.managed.disableServiceAccountKeyCreation \
    --project="$PROJECT_ID" --effective --format="value(spec.rules[0].enforce)" 2>/dev/null || echo "")
if [ "$KEY_POLICY" = "True" ]; then
    cat >&2 <<EOF
ERROR: iam.managed.disableServiceAccountKeyCreation is ENFORCED on $PROJECT_ID,
so a service-account key cannot be created here. Either pick a project with a
project-level override, or (with roles/orgpolicy.policyAdmin) add one:

    gcloud org-policies set-policy - <<'POLICY'
    name: projects/$PROJECT_ID/policies/iam.managed.disableServiceAccountKeyCreation
    spec:
      rules:
      - enforce: false
    POLICY

EOF
    exit 1
fi
echo "    service-account keys: allowed on this project"

# ---------------------------------------------------------------------------
# Vertex API
# ---------------------------------------------------------------------------
step "Ensuring the Vertex AI API is enabled"
if gcloud services list --enabled --project "$PROJECT_ID" \
        --format="value(config.name)" 2>/dev/null | grep -q "^aiplatform.googleapis.com$"; then
    echo "    already enabled"
else
    run gcloud services enable aiplatform.googleapis.com --project "$PROJECT_ID"
fi

# ---------------------------------------------------------------------------
# Service account — least privilege: aiplatform.user only
# ---------------------------------------------------------------------------
step "Creating service account $SA_EMAIL"
if gcloud iam service-accounts describe "$SA_EMAIL" --project "$PROJECT_ID" >/dev/null 2>&1; then
    echo "    already exists — reusing it"
else
    run gcloud iam service-accounts create "$SA_NAME" \
        --project "$PROJECT_ID" \
        --display-name "Vertex AI for the donor game simulations"
fi

# A freshly created service account is not immediately visible to the IAM
# policy API — add-iam-policy-binding fails with "does not exist" for a few
# seconds after create returns. Retry rather than making the caller re-run.
step "Granting roles/aiplatform.user"
if [ -n "$DRY_RUN" ]; then
    run gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member "serviceAccount:${SA_EMAIL}" \
        --role roles/aiplatform.user \
        --condition=None
else
    for attempt in 1 2 3 4 5 6; do
        if gcloud projects add-iam-policy-binding "$PROJECT_ID" \
                --member "serviceAccount:${SA_EMAIL}" \
                --role roles/aiplatform.user \
                --condition=None >/dev/null 2>&1; then
            echo "    granted (attempt $attempt)"
            break
        fi
        if [ "$attempt" = "6" ]; then
            echo "ERROR: could not bind roles/aiplatform.user after 6 attempts." >&2
            echo "Retry manually:" >&2
            echo "  gcloud projects add-iam-policy-binding $PROJECT_ID \\" >&2
            echo "    --member serviceAccount:${SA_EMAIL} \\" >&2
            echo "    --role roles/aiplatform.user --condition=None" >&2
            exit 1
        fi
        echo "    IAM not yet consistent, retrying in $((attempt * 5))s..."
        sleep $((attempt * 5))
    done
fi

step "Writing service account key to $KEY_PATH"
if [ -f "$KEY_PATH" ]; then
    echo "    key already exists — leaving it alone (delete it to mint a fresh one)"
else
    run mkdir -p "$KEY_DIR"
    run chmod 700 "$KEY_DIR"
    run gcloud iam service-accounts keys create "$KEY_PATH" \
        --iam-account "$SA_EMAIL" \
        --project "$PROJECT_ID"
    run chmod 600 "$KEY_PATH"
fi

# ---------------------------------------------------------------------------
# Optional budget, scoped to Vertex spend on this project
# ---------------------------------------------------------------------------
if [ -n "$BUDGET_AMOUNT" ]; then
    # The Budget API filters on project NUMBER, not project ID. Passing the ID
    # yields a budget that matches nothing and alerts that never fire.
    step "Creating a budget alert for Vertex spend on $PROJECT_ID"
    # Budget creation goes through the Budget API, which must be enabled on the
    # project gcloud bills the call to — separate from aiplatform.googleapis.com.
    if gcloud services list --enabled --project "$PROJECT_ID" \
            --format="value(config.name)" 2>/dev/null | grep -q "^billingbudgets.googleapis.com$"; then
        echo "    billingbudgets API already enabled"
    else
        echo "    enabling billingbudgets.googleapis.com"
        run gcloud services enable billingbudgets.googleapis.com --project "$PROJECT_ID"
    fi
    if [ -n "$DRY_RUN" ]; then
        PROJECT_NUMBER="<resolved-at-runtime>"
    else
        PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")
    fi
    # Filtering by service as well as project matters here: your-gcp-project is a
    # production project, so an unfiltered budget would alert on unrelated spend.
    run gcloud billing budgets create \
        --billing-account="$BILLING_ACCOUNT" \
        --display-name="donor-game Vertex spend (${PROJECT_ID})" \
        --budget-amount="${BUDGET_AMOUNT}USD" \
        --filter-projects="projects/${PROJECT_NUMBER}" \
        --filter-services="services/C7E2-9256-1C43" \
        --threshold-rule=percent=0.5 \
        --threshold-rule=percent=0.9 \
        --threshold-rule=percent=1.0 \
        --threshold-rule=percent=1.0,basis=forecasted-spend
    # C7E2-9256-1C43 = "Vertex AI", confirmed against the Cloud Billing Catalog
    # API. Note it is distinct from "Vertex AI Search" (74B1-77CF-C302) and
    # "Vertex AI Vision" (E5FE-878F-FECE). Re-check with:
    #   curl -H "Authorization: Bearer \$(gcloud auth print-access-token)" \
    #     https://cloudbilling.googleapis.com/v1/services?pageSize=2000
fi

cat <<EOF

======================================================================
Done. Gemini calls will bill: $PROJECT_ID
      via billing account:    $BILLING_ACCOUNT
======================================================================

1. Copy the key to the cluster:

     ssh ${CLUSTER} 'mkdir -p ~/.gcp && chmod 700 ~/.gcp'
     scp $KEY_PATH ${CLUSTER}:~/.gcp/vertex-sa.json

2. Append to \$REPO/.env ON THE CLUSTER. run_baseline_array.sh:148 does
   'set -a; source \$REPO/.env; set +a', so this is the one place that
   reliably reaches compute nodes:

     GOOGLE_GENAI_USE_VERTEXAI=true
     GOOGLE_CLOUD_PROJECT=$PROJECT_ID
     GOOGLE_CLOUD_LOCATION=$LOCATION
     GOOGLE_APPLICATION_CREDENTIALS=/home/USERNAME/.gcp/vertex-sa.json

3. Submit one task and confirm this exact line in the log:

     [gemini] Vertex AI mode via ADC — BILLING PROJECT: $PROJECT_ID

   If you instead see "AI Studio API-key mode (NOT Vertex)", the env vars
   did not reach the job and $PROJECT_ID is NOT being charged.

4. Watch spend land on the project:

     https://console.cloud.google.com/billing/$BILLING_ACCOUNT/reports?project=$PROJECT_ID

EOF
