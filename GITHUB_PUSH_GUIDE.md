# GitHub Push Guide

## Step 1: Create Repository on GitHub

1. Go to https://github.com/new
2. Fill in the details:
   - **Repository name:** `algotrader` (or your preferred name)
   - **Description:** "Quantitative trading skill for Indian equity markets with Zerodha integration"
   - **Visibility:**
     - ⚠️ **Private** (Recommended - keeps your trading strategies private)
     - Or Public (if you want to share with community)
   - **Do NOT initialize with:**
     - ❌ Don't add README (we already have one)
     - ❌ Don't add .gitignore (we already have one)
     - ❌ Don't choose a license yet

3. Click **"Create repository"**

## Step 2: Copy the Repository URL

After creating, GitHub will show you a page with setup instructions. Copy the repository URL which looks like:

**SSH (Recommended if you have SSH keys):**
```
git@github.com:YOUR_USERNAME/algotrader.git
```

**HTTPS (Easier, but requires password/token):**
```
https://github.com/YOUR_USERNAME/algotrader.git
```

## Step 3: Add Remote and Push

Once you have the URL, run these commands:

```bash
# Add the remote (replace <URL> with your actual URL)
git remote add origin <URL>

# Push to GitHub
git push -u origin main
```

## Full Example

If your GitHub username is `rakeshtechie` and you chose HTTPS:

```bash
cd /home/rakesh/work/skills/algotrader

# Add remote
git remote add origin https://github.com/rakeshtechie/algotrader.git

# Push
git push -u origin main
```

## Troubleshooting

### Issue: Authentication Failed (HTTPS)

If using HTTPS, you'll need a Personal Access Token:

1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Give it a name: "AlgoTrader CLI"
4. Select scopes: ✅ `repo` (full control of private repositories)
5. Click "Generate token"
6. **Copy the token immediately** (you won't see it again)
7. When pushing, use the token as your password

### Issue: Permission Denied (SSH)

If using SSH and you get "Permission denied":

1. Check if you have SSH keys: `ls -al ~/.ssh`
2. If not, generate: `ssh-keygen -t ed25519 -C "your_email@example.com"`
3. Add to GitHub: https://github.com/settings/keys
4. Copy public key: `cat ~/.ssh/id_ed25519.pub`
5. Paste into GitHub

### Issue: Branch Name Conflict

If GitHub says the branch should be "master" instead of "main":

```bash
git branch -M main
git push -u origin main
```

## After Pushing

Once pushed, you can:

1. View your repository at: `https://github.com/YOUR_USERNAME/algotrader`
2. Add a license (MIT recommended for open source)
3. Add topics/tags for discoverability
4. Enable GitHub Actions for CI/CD (optional)
5. Add collaborators (Settings → Manage access)

## Security Reminder

✅ Your .gitignore is already configured to exclude:
- `.env` files (API credentials)
- `venv/` (virtual environment)
- `universe/*.json` (generated data)
- Logs and backtests

⚠️ **Never commit .env files or API credentials!**

## Ready to Push?

Tell me your GitHub username, and I'll help you run the exact commands!
