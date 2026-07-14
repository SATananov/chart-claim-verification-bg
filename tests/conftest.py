import matplotlib


# Pytest does not need an interactive window.
# The Agg backend creates PNG files without Tkinter.
matplotlib.use("Agg")
