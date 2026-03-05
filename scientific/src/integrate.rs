pub fn quad(f: impl Fn(f64) -> f64, a: f64, b: f64) -> (f64, f64) {
    let n = 100;
    let h = (b - a) / n as f64;
    let mut sum = 0.5 * (f(a) + f(b));

    for i in 1..n {
        let x = a + i as f64 * h;
        sum += f(x);
    }

    (sum * h, 0.0)
}

pub fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let n = if n.is_multiple_of(2) { n } else { n + 1 };
    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);

    for i in 1..n {
        let x = a + i as f64 * h;
        let coeff = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum += coeff * f(x);
    }

    sum * h / 3.0
}

pub fn trapezoid(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let h = (b - a) / n as f64;
    let mut sum = (f(a) + f(b)) / 2.0;

    for i in 1..n {
        sum += f(a + i as f64 * h);
    }

    sum * h
}

pub fn ode_solve(
    f: impl Fn(f64, &[f64]) -> Vec<f64>,
    y0: &[f64],
    t_span: (f64, f64),
    dt: f64,
) -> Vec<(f64, Vec<f64>)> {
    let mut t = t_span.0;
    let mut y = y0.to_vec();
    let mut result = vec![(t, y.clone())];

    while t < t_span.1 {
        let k1 = f(t, &y);
        let k2 = f(t + dt / 2.0, &add_vec(&y, &scale_vec(&k1, dt / 2.0)));
        let k3 = f(t + dt / 2.0, &add_vec(&y, &scale_vec(&k2, dt / 2.0)));
        let k4 = f(t + dt, &add_vec(&y, &scale_vec(&k3, dt)));

        let dy = add_vec(
            &scale_vec(&k1, 1.0),
            &add_vec(
                &scale_vec(&k2, 2.0),
                &add_vec(&scale_vec(&k3, 2.0), &scale_vec(&k4, 1.0)),
            ),
        );

        y = add_vec(&y, &scale_vec(&dy, dt / 6.0));
        t += dt;

        result.push((t, y.clone()));

        if result.len() > 10000 {
            break;
        }
    }

    result
}

fn add_vec(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn scale_vec(a: &[f64], s: f64) -> Vec<f64> {
    a.iter().map(|x| x * s).collect()
}

pub fn cumtrapz(f: impl Fn(f64) -> f64, t: &[f64], initial: f64) -> Vec<f64> {
    let mut result = Vec::with_capacity(t.len());
    result.push(initial);
    let mut current = initial;

    for i in 1..t.len() {
        let dt = t[i] - t[i - 1];
        let avg = (f(t[i]) + f(t[i - 1])) / 2.0;
        current += avg * dt;
        result.push(current);
    }

    result
}
