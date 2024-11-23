@echo off

:: =====> set up Anaconda3 virtual environment =====
SET ENV_FILE=environment.yml
SET ENV_NAME=qkrt-cv
:: verify that conda is installed and call activate the base environment
CALL "%USERPROFILE%\Anaconda3\Scripts\activate.bat" base
IF NOT DEFINED CONDA_PREFIX (
    echo error: conda activation failed... ensure conda is installed correctly.
    exit /b 1
)
:: verify that environment.yml exists and create a new conda environment
IF NOT EXIST "%ENV_FILE%" (
    echo error: %ENV_FILE% not found.
    exit /b 1
)
echo info: creating conda environment %ENV_NAME% from %ENV_FILE%...
conda env create --file=%ENV_FILE%
:: verify that the environment was created successfully before continuing
IF %ERRORLEVEL% NEQ 0 (
    echo error: failed to create conda environment.
    exit /b 1
)
echo info: successfully created conda environment.
:: ===== set up Anaconda3 virtual environment <=====