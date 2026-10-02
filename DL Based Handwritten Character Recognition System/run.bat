@echo off
title NEUROSCRIPT - Starting Server
echo.
echo  ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗  ███████╗ ██████╗██████╗ ██╗██████╗ ████████╗
echo  ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗██╔════╝██╔════╝██╔══██╗██║██╔══██╗╚══██╔══╝
echo  ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║███████╗██║     ██████╔╝██║██████╔╝   ██║   
echo  ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║╚════██║██║     ██╔══██╗██║██╔═══╝    ██║   
echo  ██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝███████║╚██████╗██║  ██║██║██║        ██║   
echo  ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝╚═╝        ╚═╝   
echo.
echo  Deep Learning Handwritten Character Recognition System
echo  -------------------------------------------------------
echo  NOTE: First run will download ~400 MB model from HuggingFace.
echo  After that, it loads from cache instantly.
echo.
echo  Server will be available at: http://localhost:5000
echo  Press Ctrl+C to stop the server.
echo.
python app.py
pause
