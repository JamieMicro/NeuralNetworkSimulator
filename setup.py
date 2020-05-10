import cx_Freeze

executables = [cx_Freeze.Executable("VisualML.py")]

cx_Freeze.setup(name="Visual ML", options={"build_exe": {"packages":["pygame","numpy","random","time","configparser"], "includes": ["numpy"], "include_files":["data/","freesansbold.ttf","config.cfg"]}}, executables=executables)
