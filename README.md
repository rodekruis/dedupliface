# fastapi-template

Template repo for FastAPI. Includes CI/CD on Azure Web App using Github Actions. Uses [Poetry](https://python-poetry.org/) for dependency management.

### Setup

1. run `git clone https://github.com/jmargutt/fastapi-template.git`
2. change code as needed
3. add needed environment variables to `.env` file
> [!WARNING]  
> Do not store credentials/passswords/keys in the code, use the `.env` file instead.
> This file will not be pushed to the repository, as it is listed in the `.gitignore` file, so your credentials 
> won't be exposed.
4. add needed packages to `pyproject.toml`
5. run `poetry install` to install the packages
6. run `uvicorn main:app --reload` to start the server
7. go to `http://127.0.0.1:8000/docs` and test if the app runs as expected.

### Deploy to Azure with GitHub Actions

2. Create a new App Service Plan in Azure, or choose a pre-existing one.
2. Create a new Web App in Azure:
   * choose a meaningful name, e.g. `fastapi-template-jacopo`
   * select `Publish`: `Docker Container`
   * select `Operating System`: `Linux`
   * select `Region`: `West Europe`
   * select the App Service Plan you created in step 1
   * Create the Web App
3. Get a Web App Publish Profile, to deploy from GitHub
   * Go to your app service in the Azure portal. 
   * On the Overview page, select `Download publish profile`. 
   * Save the downloaded file. You'll use the contents of the file to create a GitHub secret.
1. Create a new Azure Container Registry or choose a pre-existing one.
   * Go to the registry -> `Access keys` -> Enable `Admin user` -> Copy the username and password
3. Give permissions to the Web App to access the Container Registry
   * Go to your app service in the Azure portal.
   * Under `Configuration`, update the `Application settings`:
     * `DOCKER_REGISTRY_SERVER_URL`: the URL of the Azure Container Registry, e.g. `fastapiregister.azurecr.io`
     * `DOCKER_REGISTRY_SERVER_USERNAME`: the username of the Azure Container Registry, e.g. `fastapiregister`
     * `DOCKER_REGISTRY_SERVER_PASSWORD`: the password of the Azure Container Registry
     * `WEBSITES_PORT`: 8000
> [!IMPORTANT]  
> These `Application settings` determine which environment variables are accessible by the web app. 
> If you change/add environment variables in the GitHub repository, don't forget to update the Web App `Configuration` in the Azure portal.

4. Create the GitHub secrets and variables, so that GitHub Actions can deploy to Azure
   * Go to your GitHub repository
   * Go to `Settings` -> `Secrets and variables` -> `Actions` -> `New repository secret`
   * Add the following **repository secrets**:
     * `AZURE_WEBAPP_PUBLISH_PROFILE`: the contents of the downloaded file from step 3
     * `REGISTRY_PASSWORD`: the password of the Azure Container Registry
   * Go to `Settings` -> `Secrets and variables` -> `Actions` -> `New repository variable`
   * Add the following **repository variables**:
     * `AZURE_WEBAPP_NAME`: the name of your web app, e.g. `fastapi-template-jacopo`
     * `REGISTRY_NAME`: the name of the Azure Container Registry, e.g. `fastapiregister`
5. Push a change to the repository to trigger the GitHub Actions workflow.
4. Wait 5-10 minutes, then go to `<my-api-name>.azurewebsites.net` and see the app running.
5. If the app is not running
   * go to the `Actions` tab in your GitHub repository and check the logs of the failed workflow.
   * go to the `Overview` tab of your Web App in the Azure portal and check `Deployment logs` -> `Logs`.

### Run locally

```
cp example.env .env
pip install poetry
poetry install
uvicorn main:app --reload
```
