# 📘 Instruções de Trabalho

[🏠 **Voltar ao README Principal**](README.md)


### 1. Instalar Git para Windows  
Faça download e instale o Git:  
🔗 https://github.com/git-for-windows/git/releases/download/v2.52.0.windows.1/Git-2.52.0-64-bit.exe  


### 2. Instalar extensão GitHub no VS Code  
No VS Code instale a extensão **GitHub Pull Requests and Issues**:  
🔗 https://code.visualstudio.com/docs/sourcecontrol/github  


### 3. Criar conta GitHub  
Criar conta em:  
🔗 https://github.com/signup  


### 4. Clonar o repositório  
Na extensão **GitHub** no VS Code:

1. Clique **Clone Repository**
2. Cole o URL fornecido na documentação
3. Quando pedir a pasta, criar e selecionar:


### 5. Configurar Git no VS Code  
No terminal do VS Code, executar:

```bash
git config --global user.name "O SEU NOME"
git config --global user.email "O_SEU_EMAIL_DO_GITHUB"
```

### 6. Instalar dependências
```bash
pip install -r requirements.txt
```
### 6. Instalar dependências
Se forem necessárias chaves API, será fornecido um ficheiro `.env`

# 🤝 Ligação com Hugging Face

### 1. Criar conta no Hugginh Face e criar o space

### 2. Gerar token em: https://huggingface.co/settings/tokens
- Nomes: github_push
- Permissões: Write

### 3. Adicionar o Space ao GITHub

```bash
git remote add space https://huggingface.co/spaces/USERNAME/NOME_DO_SPACE
```

### 4. Efetuar o primeiro push

```bash
git push --force space main
```
Irá ser pedido o login com as credenciais Hugging Face. Deverá ser utilizado o Username e o secret como password

### 5. Adicionar o secret como `HF_TOKEN` no github.
Após criado o secret deverá ser adicionado o yaml [deploy_to_hf](./.github/workflows/deploy_to_hf.yml) à pasta `./.github/workflows/deploy_to_hf.yml`
Criando assim uma GITHub action que executará sempre que um push main seja efetuado.

NOTA: necessário alterar a linha 26 `git clone https://user:$HF_TOKEN@huggingface.co/spaces/<Username>/<Hugging Face Space> hf_space`