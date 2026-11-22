# Madam Shama AI – setup & deployment guide (beginner friendly)

Google tools: **Gemini** (AI), **Google Sign-In**, **Firestore** (progress), **Cloud Storage** (uploads), **Cloud Run** (hosting).
Google's screens change often. If a button name differs, look for the same word on the page.

---
## PART A – Changing your code on GitHub

**Edit one file:** open the file in your repo → tap the ✏️ pencil → change it → **Commit changes** (keep "Commit directly to main").
**Replace files:** Add file → **Upload files** → pick the new `app.py` etc. → Commit. Same name = replaced.
**Create a new file:** Add file → **Create new file** → type the exact name (for example `Dockerfile`, no extension) → paste → Commit.
Files starting with a dot (`.dockerignore`, `.gitignore`) are easiest to create this way.
**Undo a mistake:** open the file → **History** → open the older version → copy it back into the file and commit.
**Never upload** passwords, API keys or the service-account JSON.

Files your repo needs: `app.py`, `requirements.txt`, `Dockerfile`, `.dockerignore`, `.gitignore`, `README.md`, `secrets.toml.example`.

---
## PART B – Google Cloud project and services

1. console.cloud.google.com → project picker → **New project** (e.g. madam-shama).
2. Billing: some services need a billing account (new accounts often get free credit). Create a **budget alert** (Billing → Budgets & alerts).
3. **Firestore:** search "Firestore" → Create database → Native mode → choose a region → Create.
4. **Cloud Storage:** search "Cloud Storage" → Create bucket → unique name → same region → keep public access prevented → Create. Remember the name.
5. **Google Sign-In:** APIs & Services → OAuth consent screen (may be "Google Auth Platform") → External → fill app name and email → Save.
   In "Testing" mode add student emails as **Test users**; Publish the app when ready for everyone.
   Then Credentials → Create credentials → **OAuth client ID** → Web application. Copy the Client ID and secret.
   (Redirect URI comes in Part C, after you know your app's address.)

---
## PART C – Deploy on Google Cloud Run (recommended, all Google)

**1. Turn on services:** search and **Enable**: Cloud Run, Cloud Build, Artifact Registry, Secret Manager.

**2. Make your secrets file.** Copy `secrets.toml.example`, fill it in. You do NOT need the `[gcp_service_account]` block on Cloud Run.
For now put a placeholder redirect URI; you'll fix it in step 6.
Secret Manager → **Create secret** → name `streamlit-secrets` → paste the whole file content → Create.

**3. Create the service from GitHub:**
Cloud Run → **Create service** → **Continuously deploy from a repository** → **Set up with Cloud Build** → connect GitHub → pick your repo and branch `main` → Build type **Dockerfile** → Save.
- Region: same as Firestore.
- Authentication: **Allow unauthenticated invocations** (the app has its own login).
- Memory: 1 GiB.
- Containers → **Volumes** → add a Secret volume: secret `streamlit-secrets`, mount path `/app/.streamlit`, file name `secrets.toml`; then in the Container's Volume Mounts add the mount.

**4. Permissions:** IAM → find the service account your Cloud Run service runs as (usually the "Compute Engine default service account") → add roles
**Cloud Datastore User**, **Storage Object Admin**, **Secret Manager Secret Accessor**. (Check role names if they differ.)

**5. First deploy:** wait for the build. Copy the service URL (ends with `run.app`).

**6. Fix login address:** in your OAuth client, add Authorized redirect URI `YOUR-URL/oauth2callback`.
Edit the secret: add a **new version** with `redirect_uri = "YOUR-URL/oauth2callback"`. Then redeploy (Cloud Run → Edit & deploy new revision → Deploy).

**7. Updates:** every time you commit to GitHub `main`, Cloud Build rebuilds and redeploys automatically (a few minutes).

Quick alternative from Cloud Shell (the `>_` icon in the console): `git clone YOUR-REPO`, `cd` into it, then
`gcloud run deploy madam-shama --source . --region YOUR-REGION --allow-unauthenticated` (secrets still need steps 2 and 4).

---
## PART D – Other option: Streamlit Community Cloud (not Google)
share.streamlit.io → connect GitHub → main file `app.py` → Advanced settings → Secrets → paste your full secrets file **including** `[gcp_service_account]` (service-account JSON key: IAM → Service Accounts → Keys → Add key → JSON).

---
## Voice typing
🎤 types your words live into the box. In Chat/Interview, **📤 Send** sends it. In other boxes, tap the box first, tap 🎤, then **✅ Done**, then press the page's own button. Works in Chrome/Edge.
