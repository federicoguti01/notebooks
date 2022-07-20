#!/usr/bin/env python
# coding: utf-8

# **Math - Differential Calculus**
# 
# Calculus is the study of continuous change. It has two major subfields: *differential calculus*, which studies the rate of change of functions, and *integral calculus*, which studies the area under the curve. In this notebook, we will discuss the former.
# 
# *Differential calculus is at the core of Deep Learning, so it is important to understand what derivatives and gradients are, how they are used in Deep Learning, and understand what their limitations are.*
# 
# **Note:** the code in this notebook is only used to create figures and animations. You do not need to understand how it works (although I did my best to make it clear, in case you are interested).

# # Slope of a straight line

# In[1]:


#@title
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# To get smooth animations
import matplotlib.animation as animation
mpl.rc('animation', html='jshtml')


# What is the slope of the following line?

# In[2]:


#@title
def get_AB_line(A_pos, B_pos, x_min=-1000, x_max=+1000):
    rise = B_pos[1] - A_pos[1]
    run = B_pos[0] - A_pos[0]
    slope = rise / run
    offset = A_pos[1] - slope * A_pos[0]
    return [x_min, x_max], [x_min * slope + offset, x_max * slope + offset]

def plot_AB_line(A_pos, B_pos, A_name="A", B_name="B"):
    for point, name in ((A_pos, A_name), (B_pos, B_name)):
        plt.plot(point[0], point[1], "bo")
        plt.text(point[0] - 0.35, point[1], name, fontsize=14)
    xs, ys = get_AB_line(A_pos, B_pos)
    plt.plot(xs, ys)

def plot_rise_over_run(A_pos, B_pos):
    plt.plot([A_pos[0], B_pos[0]], [A_pos[1], A_pos[1]], "k--")
    plt.text((A_pos[0] + B_pos[0]) / 2, A_pos[1] - 0.4, "run", fontsize=14)
    plt.plot([B_pos[0], B_pos[0]], [A_pos[1], B_pos[1]], "k--")
    plt.text(B_pos[0] + 0.2, (A_pos[1] + B_pos[1]) / 2, "rise", fontsize=14)

def show(axis="equal", ax=None, title=None, xlabel="$x$", ylabel="$y$"):
    ax = ax or plt.gca()
    ax.axis(axis)
    ax.grid()
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14, rotation=0)
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')

A_pos = np.array([1, 1])
B_pos = np.array([7, 4])
plot_AB_line(A_pos, B_pos)
plot_rise_over_run(A_pos, B_pos)
show([0, 8.4, 0, 5.5], title="Slope = rise / run")


# As you probably know, the slope of a (non-vertical) straight line can be calculated by taking any two points $\mathrm{A}$ and $\mathrm{B}$ on the line, and computing the "rise over run":
# 
# $slope = \dfrac{rise}{run} = \dfrac{\Delta y}{\Delta x} = \dfrac{y_\mathrm{B} - y_\mathrm{A}}{x_\mathrm{B} - x_\mathrm{A}}$
# 
# 
# In this example, the rise is 3, and the run is 6, so the slope is 3/6 = 0.5.

# # Defining the slope of a curve

# But what if you want to know the slope of something else than a straight line? For example, let's consider the curve defined by $y = f(x) = x^2$:

# In[3]:


#@title
xs = np.linspace(-2.1, 2.1, 500)
ys = xs**2
plt.plot(xs, ys)

plt.plot([0, 0], [0, 3], "k--")
plt.arrow(-1.4, 2.5, 0.5, -1.3, head_width=0.1)
plt.arrow(0.85, 1.05, 0.5, 1.3, head_width=0.1)
show([-2.1, 2.1, 0, 2.8], title="Slope of the curve $y = x^2$")


# Obviously, the slope varies: on the left (i.e., when $x<0$), the slope is negative (i.e., when we move from left to right, the curve goes down), while on the right (i.e., when $x>0$) the slope is positive (i.e., when we move from left to right, the curve goes up). At the point $x=0$, the slope is equal to 0 (i.e., the curve is locally flat). The fact that the slope is 0 when we reach a minimum (or indeed a maximum) is crucially important, and we will come back to it later.

# How can we put numbers on these intuitions? Well, say we want to estimate the slope of the curve at a point $\mathrm{A}$, we can do this by taking another point $\mathrm{B}$ on the curve, not too far away, and then computing the slope between these two points:
# 

# In[4]:


#@title
def animate_AB_line(f, fp, f_str, x_A, axis=None):
    y_A = f(x_A)
    eps = 1e-4
    x_B_range = 1.5
    x_B = x_A + eps

    n_frames = 200
    text_offset_A = -0.2
    text_offset_B = +0.1
    x_min, x_max = -1000, 1000

    fig, ax = plt.subplots()

    # plot f(x)
    xs = np.linspace(-2.1, 2.1, 500)
    ys = f(xs)
    ax.plot(xs, ys)

    # plot the tangent to the curve at point A
    if fp:
        slope = fp(x_A)
        offset = y_A - slope * x_A
        ax.plot([x_min, x_max], [slope*x_min + offset, slope*x_max + offset],
                "y--")

    # plot the line AB and the labels A and B so they can be animated
    y_A = f(x_A)
    y_B = f(x_B)
    xs, ys = get_AB_line([x_A, y_A], [x_B, y_B])
    line_inf, = ax.plot(xs, ys, "-")
    line_AB, = ax.plot([x_A, x_B], [y_A, y_B], "bo-")
    ax.text(x_A + text_offset_A, y_A, "A", fontsize=14)
    B_text = ax.text(x_B + text_offset_B, y_B, "B", fontsize=14)

    # plot the grid and axis labels
    title = r"Slope of the curve $y = {}$ at $x_\mathrm{{A}} = {}$".format(f_str, x_A)
    show(axis or [-2.1, 2.1, 0, 2.8], title=title)

    def update_graph(i):
        x_B = x_A + x_B_range * np.cos(i * 2 * np.pi / n_frames) ** 3
        if np.abs(x_B - x_A) < eps:
            x_B = x_A + eps # to avoid division by 0
        y_B = f(x_B)
        xs, ys = get_AB_line([x_A, y_A], [x_B, y_B])
        line_inf.set_data(xs, ys)
        line_AB.set_data([x_A, x_B], [y_A, y_B])
        B_text.set_position([x_B + text_offset_B, y_B])
        return line_inf, line_AB

    anim = animation.FuncAnimation(fig, update_graph,
                                  init_func=lambda: update_graph(0),
                                  frames=n_frames,
                                  interval=20,
                                  blit=True)
    plt.close()
    return anim

animate_AB_line(lambda x: x**2, lambda x: 2*x, "x^2", -1)


# As you can see, when point $\mathrm{B}$ is very close to point $\mathrm{A}$, the $(\mathrm{AB})$ line becomes almost indistinguishable from the curve itself (at least locally around point $\mathrm{A}$). The $(\mathrm{AB})$ line gets closer and closer to the **tangent** line to the curve at point $\mathrm{A}$: this is the best linear approximation of the curve at point $\mathrm{A}$.
# 
# So it makes sense to define the slope of the curve at point $\mathrm{A}$ as the slope that the $\mathrm{(AB)}$ line approaches when $\mathrm{B}$ gets infinitely close to $\mathrm{A}$. This slope is called the **derivative** of the function $f$ at $x=x_\mathrm{A}$. For example, the derivative of the function $f(x)=x^2$ at $x=x_\mathrm{A}$ is equal to $2x_\mathrm{A}$ (we will see how to get this result shortly), so on the graph above, since the point $\mathrm{A}$ is located at $x_\mathrm{A}=-1$, the tangent line to the curve at that point has a slope of $-2$.

# # Differentiability
# 
# Note that some functions are not quite as well-behaved as $x^2$: for example, consider the function $f(x)=|x|$, the absolute value of $x$:

# In[5]:


#@title
animate_AB_line(lambda x: np.abs(x), None, "|x|", 0)


# No matter how much you zoom in on the origin (the point at $x=0, y=0$), the curve will always look like a V. The slope is -1 for any $x < 0$, and it is +1 for any $x > 0$, but **at $x = 0$, the slope is undefined**, since it is not possible to approximate the curve $y=|x|$ locally around the origin using a straight line, no matter how much you zoom in on that point.
# 
# The function $f(x)=|x|$ is said to be **non-differentiable** at $x=0$: its derivative is undefined at $x=0$. This means that the curve $y=|x|$ has an undefined slope at that point. However, the function $f(x)=|x|$ is **differentiable** at all other points.
# 
# In order for a function $f(x)$ to be differentiable at some point $x_\mathrm{A}$, the slope of the $(\mathrm{AB})$ line must approach a single finite value as $\mathrm{B}$ gets infinitely close to $\mathrm{A}$.
# 
# This implies several constraints:
# 
# * First, the function must of course be **defined** at $x_\mathrm{A}$. As a counterexample, the function $f(x)=\dfrac{1}{x}$ is undefined at $x_\mathrm{A}=0$, so it is not differentiable at that point.
# * The function must also be **continuous** at $x_\mathrm{A}$, meaning that as $x_\mathrm{B}$ gets infinitely close to $x_\mathrm{A}$, $f(x_\mathrm{B})$ must also get infinitely close to $f(x_\mathrm{A})$. As a counterexample, $f(x)=\begin{cases}-1 \text{ if }x < 0\\+1 \text{ if }x \geq 0\end{cases}$ is not continuous at $x_\mathrm{A}=0$, even though it is defined at that point: indeed, when you approach it from the negative side, it does not approach infinitely close to $f(0)=+1$. Therefore, it is not continuous at that point, and thus not differentiable either.
# * The function must not have a **breaking point** at $x_\mathrm{A}$, meaning that the slope that the $(\mathrm{AB})$ line approaches as $\mathrm{B}$ approaches $\mathrm{A}$ must be the same whether $\mathrm{B}$ approaches from the left side or from the right side. We already saw a counterexample with $f(x)=|x|$, which is both defined and continuous at $x_\mathrm{A}=0$, but which has a breaking point at $x_\mathrm{A}=0$: the slope of the curve $y=|x|$ is -1 on the left, and +1 on the right.
# * The curve $y=f(x)$ must not be **vertical** at point $\mathrm{A}$. One counterexample is $f(x)=\sqrt[3]{x}$, the cubic root of $x$: the curve is vertical at the origin, so the function is not differentiable at $x_\mathrm{A}=0$, as you can see in the following animation:

# In[6]:


#@title
animate_AB_line(lambda x: np.cbrt(x), None, r"\sqrt[3]{x}", 0,
                axis=[-2.1, 2.1, -1.4, 1.4])


# Now let's see how to actually differentiate a function (i.e., find its derivative).

# # Differentiating a function

# The previous discussion leads to the following definition:

# <hr />
# 
# The **derivative** of a function $f(x)$ at $x = x_\mathrm{A}$ is noted $f'(x_\mathrm{A})$, and it is defined as:
# 
# $f'(x_\mathrm{A}) = \underset{x_\mathrm{B} \to x_\mathrm{A}}\lim\dfrac{f(x_\mathrm{B}) - f(x_\mathrm{A})}{x_\mathrm{B} - x_\mathrm{A}}$
# 
# <hr />

# Don't be scared, this is simpler than it looks! You may recognize the _rise over run_ equation $\dfrac{y_\mathrm{B} - y_\mathrm{A}}{x_\mathrm{B} - x_\mathrm{A}}$ that we discussed earlier. That's just the slope of the $\mathrm{(AB)}$ line. And the notation $\underset{x_\mathrm{B} \to x_\mathrm{A}}\lim$ means that we are making $x_\mathrm{B}$ approach infinitely close to $x_\mathrm{A}$. So in plain English, $f'(x_\mathrm{A})$ is the value that the slope of the $\mathrm{(AB)}$ line approaches when $\mathrm{B}$ gets infinitely close to $\mathrm{A}$. This is just a formal way of saying exactly the same thing as earlier.

# ## Example: finding the derivative of $x^2$

# Let's look at a concrete example. Let's see if we can determine what the slope of the $y=x^2$ curve is, at any point $\mathrm{A}$ (try to understand each line, I promise it's not that hard):
# 
# $
# \begin{align*}
# f'(x_\mathrm{A}) \, & = \underset{x_\mathrm{B} \to x_\mathrm{A}}\lim\dfrac{f(x_\mathrm{B}) - f(x_\mathrm{A})}{x_\mathrm{B} - x_\mathrm{A}} \\
# & = \underset{x_\mathrm{B} \to x_\mathrm{A}}\lim\dfrac{{x_\mathrm{B}}^2 - {x_\mathrm{A}}^2}{x_\mathrm{B} - x_\mathrm{A}} \quad && \text{since } f(x) = x^2\\
# & = \underset{x_\mathrm{B} \to x_\mathrm{A}}\lim\dfrac{(x_\mathrm{B} - x_\mathrm{A})(x_\mathrm{B} + x_\mathrm{A})}{x_\mathrm{B} - x_\mathrm{A}}\quad && \text{since } {x_\mathrm{A}}^2 - {x_\mathrm{B}}^2 = (x_\mathrm{A}-x_\mathrm{B})(x_\mathrm{A}+x_\mathrm{B})\\
# & = \underset{x_\mathrm{B} \to x_\mathrm{A}}\lim(x_\mathrm{B} + x_\mathrm{A})\quad && \text{since the two } (x_\mathrm{B} - x_\mathrm{A}) \text{ cancel out}\\
# & = \underset{x_\mathrm{B} \to x_\mathrm{A}}\lim x_\mathrm{B} \, + \underset{x_\mathrm{B} \to x_\mathrm{A}}\lim x_\mathrm{A}\quad && \text{since the limit of a sum is the sum of the limits}\\
# & = x_\mathrm{A} \, + \underset{x_\mathrm{B} \to x_\mathrm{A}}\lim x_\mathrm{A} \quad && \text{since } x_\mathrm{B}\text{ approaches } x_\mathrm{A} \\
# & = x_\mathrm{A} + x_\mathrm{A} \quad && \text{since } x_\mathrm{A} \text{ remains constant when } x_\mathrm{B}\text{ approaches } x_\mathrm{A} \\
# & = 2 x_\mathrm{A}
# \end{align*}
# $
# 
# That's it! We just proved that the slope of $y = x^2$ at any point $\mathrm{A}$ is $f'(x_\mathrm{A}) = 2x_\mathrm{A}$. What we have done is called **differentiation**: finding the derivative of a function.

# Note that we used a couple of important properties of limits. Here are the main properties you need to know to work with derivatives:
# 
# * $\underset{x \to k}\lim c = c \quad$ if $c$ is some constant value that does not depend on $x$, then the limit is just $c$.
# * $\underset{x \to k}\lim x = k \quad$ if $x$ approaches some value $k$, then the limit is $k$.
# * $\underset{x \to k}\lim\,\left[f(x) + g(x)\right] = \underset{x \to k}\lim f(x) + \underset{x \to k}\lim g(x) \quad$ the limit of a sum is the sum of the limits
# * $\underset{x \to k}\lim\,\left[f(x) \times g(x)\right] = \underset{x \to k}\lim f(x) \times \underset{x \to k}\lim g(x) \quad$ the limit of a product is the product of the limits
# 

# **Important note:** in Deep Learning, differentiation is almost always performed automatically by the framework you are using (such as TensorFlow or PyTorch). This is called auto-diff, and I did [another notebook](https://github.com/ageron/handson-ml2/blob/master/extra_autodiff.ipynb) on that topic. However, you should still make sure you have a good understanding of derivatives, or else they will come and bite you one day, for example when you use a square root in your cost function without realizing that its derivative approaches infinity when $x$ approaches 0 (tip: you should use $\sqrt{x+\epsilon}$ instead, where $\epsilon$ is some small constant, such as $10^{-4}$).

# You will often find a slightly different (but equivalent) definition of the derivative. Let's derive it from the previous definition. First, let's define $\epsilon = x_\mathrm{B} - x_\mathrm{A}$. Next, note that $\epsilon$ will approach 0 as $x_\mathrm{B}$ approaches $x_\mathrm{A}$. Lastly, note that $x_\mathrm{B} = x_\mathrm{A} + \epsilon$. With that, we can reformulate the definition above like so:
# 
# $f'(x_\mathrm{A}) = \underset{\epsilon \to 0}\lim\dfrac{f(x_\mathrm{A} + \epsilon) - f(x_\mathrm{A})}{\epsilon}$

# While we're at it, let's just rename $x_\mathrm{A}$ to $x$, to get rid of the annoying subscript A and make the equation simpler to read:
# 
# <hr />
# 
# $f'(x) = \underset{\epsilon \to 0}\lim\dfrac{f(x + \epsilon) - f(x)}{\epsilon}$
# 
# <hr />

# Okay! Now let's use this new definition to find the derivative of $f(x) = x^2$ at any point $x$, and (hopefully) we should find the same result as above (except using $x$ instead of $x_\mathrm{A}$):
# 
# $
# \begin{align*}
# f'(x) \, & = \underset{\epsilon \to 0}\lim\dfrac{f(x + \epsilon) - f(x)}{\epsilon} \\
# & = \underset{\epsilon \to 0}\lim\dfrac{(x + \epsilon)^2 - {x}^2}{\epsilon} \quad && \text{since } f(x) = x^2\\
# & = \underset{\epsilon \to 0}\lim\dfrac{{x}^2 + 2x\epsilon + \epsilon^2 - {x}^2}{\epsilon}\quad && \text{since } (x + \epsilon)^2 = {x}^2 + 2x\epsilon + \epsilon^2\\
# & = \underset{\epsilon \to 0}\lim\dfrac{2x\epsilon + \epsilon^2}{\epsilon}\quad && \text{since the two } {x}^2 \text{ cancel out}\\
# & = \underset{\epsilon \to 0}\lim \, (2x + \epsilon)\quad && \text{since } 2x\epsilon \text{ and } \epsilon^2 \text{ can both be divided by } \epsilon\\
# & = 2 x
# \end{align*}
# $
# 
# Yep! It works out.

# ## Notations

# A word about notations: there are several other notations for the derivative that you will find in the literature:
# 
# $f'(x) = \dfrac{\mathrm{d}f(x)}{\mathrm{d}x} = \dfrac{\mathrm{d}}{\mathrm{d}x}f(x)$
# 
# This notation is also handy when a function is not named. For example $\dfrac{\mathrm{d}}{\mathrm{d}x}[x^2]$ refers to the derivative of the function $x \mapsto x^2$.
# 
# Moreover, when people talk about the function $f(x)$, they sometimes leave out "$(x)$", and they just talk about the function $f$. When this is the case, the notation of the derivative is also simpler:
# 
# $f' = \dfrac{\mathrm{d}f}{\mathrm{d}x} = \dfrac{\mathrm{d}}{\mathrm{d}x}f$
# 
# The $f'$ notation is Lagrange's notation, while $\dfrac{\mathrm{d}f}{\mathrm{d}x}$ is Leibniz's notation.
# 
# There are also other less common notations, such as Newton's notation $\dot y$ (assuming $y = f(x)$) or Euler's notation $\mathrm{D}f$.

# ## Plotting the tangent to a curve

# Let's use the equation $f'(x) = 2x$ to plot the tangent to the $y=x^2$ curve at various values of $x$ (you can click on the play button under the graphs to play the animation):

# In[7]:


#@title
def animate_tangent(f, fp, f_str):
    n_frames = 200
    x_min, x_max = -1000, 1000

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8), sharex=True)

    # plot f
    xs = np.linspace(-2.1, 2.1, 500)
    ys = f(xs)
    ax1.plot(xs, ys)

    # plot tangent
    line_tangent, = ax1.plot([x_min, x_max], [0, 0])

    # plot f'
    xs = np.linspace(-2.1, 2.1, 500)
    ys = fp(xs)
    ax2.plot(xs, ys, "r-")

    # plot points A
    point_A1, = ax1.plot(0, 0, "bo")
    point_A2, = ax2.plot(0, 0, "bo")

    show([-2.1, 2.1, 0, 2.8], ax=ax1, ylabel="$f(x)$",
        title=r"$y=f(x)=" + f_str + "$ and the tangent at $x=x_\mathrm{A}$")
    show([-2.1, 2.1, -4.2, 4.2], ax=ax2, ylabel="$f'(x)$",
        title=r"y=f'(x) and the slope of the tangent at $x=x_\mathrm{A}$")

    def update_graph(i):
        x = 1.5 * np.sin(2 * np.pi * i / n_frames)
        f_x = f(x)
        df_dx = fp(x)
        offset = f_x - df_dx * x
        line_tangent.set_data([x_min, x_max],
                              [df_dx * x_min + offset, df_dx * x_max + offset])
        point_A1.set_data(x, f_x)
        point_A2.set_data(x, df_dx)
        return line_tangent, point_A1, point_A2

    anim = animation.FuncAnimation(fig, update_graph,
                                  init_func=lambda: update_graph(0),
                                  frames=n_frames,
                                  interval=20,
                                  blit=True)
    plt.close()
    return anim

def f(x):
  return x**2

def fp(x):
  return 2*x

animate_tangent(f, fp, "x^2")

