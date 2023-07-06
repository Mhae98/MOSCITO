"""Code based on https://stackoverflow.com/questions/44985966/managing-dynamic-plotting-in-matplotlib-animation
-module/44989063#44989063 """

from matplotlib.animation import FuncAnimation
import matplotlib.widgets
from mpl_toolkits import axes_grid1


class AnimatedPlotter(FuncAnimation):
    """
    Animated plotter with media playback buttons
    """
    def __init__(self, fig, function, init_func=None, func_args=None, save_count=None, min_i=0,
                 max_i=100, pos=(0.150, 0.95), interval=100, **kwargs):
        self.i = 0
        self.min_i = min_i
        self.max_i = max_i
        self.forwards = True
        self.fig = fig
        self.func = function
        self.func_args = func_args
        player_ax = self.fig.add_axes([pos[0], pos[1], 0.64, 0.04])
        divider = axes_grid1.make_axes_locatable(player_ax)
        back_ax = divider.append_axes("right", size="80%", pad=0.05)
        stop_ax = divider.append_axes("right", size="80%", pad=0.05)
        forward_ax = divider.append_axes("right", size="80%", pad=0.05)
        one_forward_ax = divider.append_axes("right", size="100%", pad=0.05)
        slider_ax = divider.append_axes("right", size="500%", pad=0.2)
        self.button_one_back = matplotlib.widgets.Button(player_ax, label='$\u29CF$')
        self.button_back = matplotlib.widgets.Button(back_ax, label='$\u25C0$')
        self.button_stop = matplotlib.widgets.Button(stop_ax, label='$\u25A0$')
        self.button_forward = matplotlib.widgets.Button(forward_ax, label='$\u25B6$')
        self.button_one_forward = matplotlib.widgets.Button(one_forward_ax, label='$\u29D0$')
        self.slider = matplotlib.widgets.Slider(slider_ax, '', self.min_i, self.max_i, valinit=self.i)
        self.setup_widgets()

        FuncAnimation.__init__(self, self.fig, self.func, frames=self.play(), init_func=init_func,
                               fargs=func_args, save_count=save_count, interval=interval, repeat=True)

    def play(self) -> int:
        """Runs the animation"""
        while True:
            self.i = self.i + self.forwards - (not self.forwards)
            self.update_slider()
            if self.min_i < self.i < self.max_i:
                yield self.i
            elif self.min_i > self.i:
                self.i = self.min_i
                self.update_slider()
                self.stop()
                yield self.i
            elif self.i > self.max_i:
                self.i = self.max_i
                self.update_slider()
                self.stop()
                yield self.i
            else:
                self.stop()
                yield self.i

    def run(self):
        """Resumes running the animation"""
        self.resume()

    def stop(self, event=None):
        """Handles click on pause button and stops the animation"""
        self.pause()

    def forward(self, event=None):
        """Runs the animation in forward direction"""
        if not self.forwards:
            self.pause()
        self.forwards = True
        self.run()

    def backward(self, event=None):
        """Runs the animation in backwards direction"""
        self.forwards = False
        self.run()

    def one_forward(self, event=None):
        """Progress animation by one step in forward direction"""
        self.forwards = True
        self.one_step()

    def one_backward(self, event=None):
        """Progress animation by one step in backwards direction"""
        self.forwards = False
        self.one_step()

    def one_step(self):
        """Progress animation by one step in the direction specified by self.forwards"""
        if self.min_i < self.i < self.max_i:
            self.i = self.i + self.forwards - (not self.forwards)
        elif self.i == self.min_i and self.forwards:
            self.i += 1
        elif self.i == self.max_i and not self.forwards:
            self.i -= 1
        self.func(self.i, *self.func_args)
        self.fig.canvas.draw_idle()
        self.update_slider()

    def set_pos(self, i):
        """Set animation index to value of slider"""
        self.i = int(self.slider.val)
        self.func(self.i, *self.func_args)

    def update_slider(self):
        """Set slider to current index"""
        self.slider.set_val(self.i)

    def setup_widgets(self):
        """Set handlers of the widgets"""
        self.button_one_back.on_clicked(self.one_backward)
        self.button_back.on_clicked(self.backward)
        self.button_stop.on_clicked(self.stop)
        self.button_forward.on_clicked(self.forward)
        self.button_one_forward.on_clicked(self.one_forward)
        self.slider.on_changed(self.set_pos)
