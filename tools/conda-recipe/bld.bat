if not defined APPVEYOR (
    echo not on appveyor
   "%PYTHON%" setup.py install
) else ( 
    echo on appveyor
    cmd /E:ON /V:ON /C %APPVEYOR_BUILD_FOLDER%\tools\ci\appveyor\run_with_env.cmd "%PYTHON%" setup.py install
)
if errorlevel 1 exit 1
