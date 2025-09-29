from main import PhysicsExperimentBasic

pe = PhysicsExperimentBasic()

pe.read_csv("data2.csv")

pe.sinusoidal(function_type='basic')
pe.plot(save=False, file_path="testfig4.png", show_equation=False, show_stat=True, show_ui=True)