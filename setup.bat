@echo off
title Raytron SAM3 一键安装

echo ============================================
echo   Raytron SAM3 推理环境一键安装
echo ============================================
echo.

:: 检查 Docker
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 Docker，请先安装 Docker Desktop
    echo   下载地址: https://www.docker.com/products/docker-desktop/
    pause
    exit /b 1
)
echo [✓] Docker 已安装

:: 拉取基础镜像
echo.
echo [1/2] 拉取基础镜像 supervisely/sam3:1.0.6（首次约 12GB）...
docker pull supervisely/sam3:1.0.6
if %errorlevel% neq 0 (
    echo [错误] 基础镜像拉取失败，请检查网络或 Docker Desktop 状态
    pause
    exit /b 1
)

:: 构建项目镜像
echo.
echo [2/2] 构建项目镜像...
docker build -t raytron-prompt .
if %errorlevel% neq 0 (
    echo [错误] 构建失败，请确认当前目录下有 Dockerfile
    pause
    exit /b 1
)

echo.
echo ============================================
echo   安装完成！
echo ============================================
echo.
echo   启动命令（在项目根目录执行）:
echo.
echo     docker run --gpus all -it -v ${PWD}\model:/raytron/model -v ${PWD}\test:/raytron/test raytron-prompt
echo.
echo   首次运行前请确保:
echo     - model/sam3.pt 已就位
echo     - test/data1~7 已就位
echo     - test/json/ 下有任务 JSON
echo.
pause
