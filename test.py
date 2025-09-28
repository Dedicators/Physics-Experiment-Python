from main import PhysicsExperimentBasic

pe = PhysicsExperimentBasic()

pe.read_csv("data2.csv")

pe.polynomial(2)
pe.plot(save=True, file_path="testfig2.png", show_equation=False, show_stat=True)