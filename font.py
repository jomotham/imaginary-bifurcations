import matplotlib.pyplot as plt
import matplotlib

font_list = sorted(matplotlib.font_manager.get_font_names())

fig, ax = plt.subplots()

for i, item in enumerate(font_list):
    plt.annotate(item, (5*(i % 25),0.5*i), fontname = item)

ax.set_ylim(0,125)
ax.set_xlim(0,50)
plt.show()

