# Render Deployment Guide

## 1. Push project to GitHub

Create a new GitHub repository and push the project:

```bash
git init
git add .
git commit -m "initial commit"
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Make sure the following are in `.gitignore` and are NOT committed:

```
.env
__pycache__/
.pytest_cache/
sessions/*.json
cache/*.pkl
```

---

## 2. Create a new Web Service on Render

1. Go to https://dashboard.render.com and click **New > Web Service**.
2. Connect your GitHub account and select the repository.
3. Choose the **Free** or **Starter** plan.

---

## 3. Build & Start commands

| Setting | Value |
|---|---|
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn app:app` |

Render automatically runs the build command on every deploy.

---

## 4. Required environment variables

Set these in **Render > Your Service > Environment**:

| Variable | Value |
|---|---|
| `GEMINI_API_KEY` | Your Google Gemini API key |
| `FLASK_SECRET_KEY` | A long random string (e.g. `openssl rand -hex 32`) |
| `FLASK_DEBUG` | `false` |

---

## 5. Verify after deployment

**Static files:**
- Visit `https://<your-app>.onrender.com/` — page should load with RTL Hebrew layout.
- Check browser DevTools Network tab: `static/css/style.css` and `static/js/app.js` should return HTTP 200.

**API:**
```bash
curl -X POST https://<your-app>.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "שלום, יש לי דירת 2 חדרים בתל אביב"}'
```
Expect HTTP 200 with a JSON `{"reply": "...", "session_id": "..."}`.

**Logs:**
- Render dashboard > Your Service > **Logs** tab.
- On first boot, look for lines like:
  ```
  INFO data.loader: City 'amsterdam' ready.
  INFO data.loader: City 'london' ready.
  INFO data.loader: City 'rome' ready.
  ```

---

## 6. passenger_wsgi.py

`passenger_wsgi.py` is the cPanel/Phusion Passenger entry point. It can remain in the repository — Render does not use it. Render uses only the **Start Command** (`gunicorn app:app`).

---

## 7. Files that must NOT be committed

| Path | Reason |
|---|---|
| `.env` | Contains secret keys |
| `__pycache__/` | Python bytecode, auto-generated |
| `.pytest_cache/` | Test runner cache |
| `sessions/*.json` | Per-user session data |
| `cache/*.pkl` | KMeans model cache (rebuilt on first boot) |
