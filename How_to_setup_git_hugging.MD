## Intruções trabalho
1. Fazer download git para windows: https://github.com/git-for-windows/git/releases/download/v2.52.0.windows.1/Git-2.52.0-64-bit.exe
2. Instalar a extensão de git https://code.visualstudio.com/docs/sourcecontrol/github
3. Criar conta github
4. Na extensão fazer clone repository e por o url como na documentação
    - Quando for pedido um path para guardar os repositórios criar `C:\GIT` e apontar essa pasta
5. Abrir o terminal no vscode e adicionar o seguinte código, editando o nome e o e-mail (do github):
    ```bash
    git config --global user.name "YOUR_NAME"
    git config --global user.email "YOUR_EMAIL"
6. no terminal correr
    ```bash
    pip install -r requirements.txt
7. Se forem necessárias API KEY pedir .env