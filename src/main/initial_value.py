def euler(dy, a, b, n, alpha):
    h = (b - a) / n
    t = a
    w = alpha

    for i in range(1, n + 1):
        w += h * dy(t, w)
        t = a + i*h

    return w


def runge_kutta(dy, a, b, n, alpha):
    h = (b - a) / n
    t = a
    w = alpha

    for i in range(1, n + 1):
        k1 = h * dy(t, w)
        k2 = h * dy(t + h/2, w + k1/2)
        k3 = h * dy(t + h/2, w + k2/2)
        k4 = h * dy(t + h, w + k3)

        w += (k1 + 2*k2 + 2*k3 + k4) / 6
        t = a + i*h

    return w
