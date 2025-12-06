# 📘 Work Instructions

[🏠 Back to Main README](README.MD)

---

## 1. Install Git for Windows
Download and install Git:  
https://github.com/git-for-windows/git/releases/download/v2.52.0.windows.1/Git-2.52.0-64-bit.exe

---

## 2. Install GitHub Extension in VS Code
In VS Code, install the extension **GitHub Pull Requests and Issues**:  
https://code.visualstudio.com/docs/sourcecontrol/github

---

## 3. Create a GitHub Account
https://github.com/signup

---

## 4. Clone the Repository
In the GitHub panel inside VS Code:

1. Click **Clone Repository**
2. Paste the URL provided in the documentation
3. When prompted for a folder, create and select one

---

## 5. Configure Git in VS Code

```bash
git config --global user.name "YOUR NAME"
git config --global user.email "YOUR_GITHUB_EMAIL"
```

## 6. Install Dependencies

```bash
pip install -r requirements.txt
```

### 7. Environment Variables
If API keys are required for the project, a `.env` file will be provided with the necessary credentials.

# 🤝 Connecting Your Project to Hugging Face

### 1. Create a Hugging Face Account and Space
Log in or create an account at: https://huggingface.co
Then create a new Space for hosting your Gradio application.

### 2. Generate an Access Token: https://huggingface.co/settings/tokens
- Name: github_push
- Permissions: Write

### 3. Add the Hugging Face Space as *Remote* in Git

```bash
git remote add space https://huggingface.co/spaces/USERNAME/NOME_DO_SPACE
```
Replace `USERNAME` and `SPACE_NAME` with your actual Hugging Face username and space name.

### 4. Perform the First Push to Hugging Face

```bash
git push --force space main
```
When prompted:
- Username: your Hugging Face username
- Password: the access token you created

### 5. Add the Token as `HF_TOKEN` in GitHub
1. Go to your GitHub repository
2. Navigate to Settings → Secrets and Variables → Actions
3. Create a new secret named: HF_TOKEN
4. Paste your Hugging Face token

Next, create the workflow file:
```bash
./.github/workflows/deploy_to_hf.yml
```

This workflow ensures that every push to main automatically deploys the updated app to Hugging Face.

**IMPORTANT**:
Update line 26 of the YAML file: git clone `https://user:$HF_TOKEN@huggingface.co/spaces/<Username>/<Hugging Face Space> hf_space`

Replace `<Username>` and `<Hugging Face Space>` with your actual values.