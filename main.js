// Simple scroll reveal animation
document.addEventListener('DOMContentLoaded', () => {
    const sections = document.querySelectorAll('.section');

    const observerOptions = {
        root: null,
        threshold: 0.1,
        rootMargin: "0px"
    };

    const observer = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    sections.forEach(section => {
        section.style.opacity = '0';
        section.style.transform = 'translateY(20px)';
        section.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
        observer.observe(section);
    });

    // Add visible class style dynamically or in CSS
    const style = document.createElement('style');
    style.innerHTML = `
        .section.visible {
            opacity: 1 !important;
            transform: translateY(0) !important;
        }
    `;
    document.head.appendChild(style);

    // Fluid Particle Animation
    initFluidParticleAnimation();

    // Image Fade Transition for Neuromorphic project
    initImageFadeTransition();

    // Animated Logo Intro Trigger
    setTimeout(() => {
        document.body.classList.add('loaded');
    }, 800); // Delay to show "Logo" state
});

function initFluidParticleAnimation() {
    const canvas = document.getElementById('qft-background');
    const ctx = canvas.getContext('2d');

    let width, height;

    // Fluid Config
    const N = 64; // Grid size
    const iter = 4;
    const diffusion = 0.0001;
    const viscosity = 0.0001;
    const dt = 0.1;

    // Vector Field Config
    const numParticles = 6000; // Dense field
    let particles = [];

    class FluidGrid {
        constructor(size, diffusion, viscosity, dt) {
            this.size = size;
            this.diff = diffusion;
            this.visc = viscosity;
            this.dt = dt;

            this.Vx = new Array(size * size).fill(0);
            this.Vy = new Array(size * size).fill(0);
            this.Vx0 = new Array(size * size).fill(0);
            this.Vy0 = new Array(size * size).fill(0);
        }

        step() {
            const N = this.size;
            const visc = this.visc;
            const dt = this.dt;
            const Vx = this.Vx;
            const Vy = this.Vy;
            const Vx0 = this.Vx0;
            const Vy0 = this.Vy0;

            diffuse(1, Vx0, Vx, visc, dt, iter, N);
            diffuse(2, Vy0, Vy, visc, dt, iter, N);

            project(Vx0, Vy0, Vx, Vy, iter, N);

            advect(1, Vx, Vx0, Vx0, Vy0, dt, N);
            advect(2, Vy, Vy0, Vx0, Vy0, dt, N);

            project(Vx, Vy, Vx0, Vy0, iter, N);
        }

        addForce(x, y, dx, dy) {
            const index = IX(x, y, this.size);
            this.Vx[index] += dx;
            this.Vy[index] += dy;
        }

        getVelocity(x, y) {
            // Bilinear interpolation
            const N = this.size;
            if (x < 0) x = 0; if (x > N - 1.001) x = N - 1.001;
            if (y < 0) y = 0; if (y > N - 1.001) y = N - 1.001;

            const i0 = Math.floor(x);
            const i1 = i0 + 1;
            const j0 = Math.floor(y);
            const j1 = j0 + 1;

            const s1 = x - i0;
            const s0 = 1.0 - s1;
            const t1 = y - j0;
            const t0 = 1.0 - t1;

            const vx = s0 * (t0 * this.Vx[IX(i0, j0, N)] + t1 * this.Vx[IX(i0, j1, N)]) +
                s1 * (t0 * this.Vx[IX(i1, j0, N)] + t1 * this.Vx[IX(i1, j1, N)]);

            const vy = s0 * (t0 * this.Vy[IX(i0, j0, N)] + t1 * this.Vy[IX(i0, j1, N)]) +
                s1 * (t0 * this.Vy[IX(i1, j0, N)] + t1 * this.Vy[IX(i1, j1, N)]);

            return { x: vx, y: vy };
        }
    }

    // Solver Helpers
    function IX(x, y, N) {
        if (x < 0) x = 0; if (x > N - 1) x = N - 1;
        if (y < 0) y = 0; if (y > N - 1) y = N - 1;
        return x + y * N;
    }

    function diffuse(b, x, x0, diff, dt, iter, N) {
        const a = dt * diff * (N - 2) * (N - 2);
        lin_solve(b, x, x0, a, 1 + 6 * a, iter, N);
    }

    function lin_solve(b, x, x0, a, c, iter, N) {
        const cRecip = 1.0 / c;
        for (let k = 0; k < iter; k++) {
            for (let j = 1; j < N - 1; j++) {
                for (let i = 1; i < N - 1; i++) {
                    x[IX(i, j, N)] =
                        (x0[IX(i, j, N)] +
                            a * (x[IX(i + 1, j, N)] +
                                x[IX(i - 1, j, N)] +
                                x[IX(i, j + 1, N)] +
                                x[IX(i, j - 1, N)]
                            )) * cRecip;
                }
            }
            set_bnd(b, x, N);
        }
    }

    function project(velocX, velocY, p, div, iter, N) {
        for (let j = 1; j < N - 1; j++) {
            for (let i = 1; i < N - 1; i++) {
                div[IX(i, j, N)] = -0.5 * (
                    velocX[IX(i + 1, j, N)] - velocX[IX(i - 1, j, N)] +
                    velocY[IX(i, j + 1, N)] - velocY[IX(i, j - 1, N)]
                ) / N;
                p[IX(i, j, N)] = 0;
            }
        }
        set_bnd(0, div, N);
        set_bnd(0, p, N);
        lin_solve(0, p, div, 1, 4, iter, N);

        for (let j = 1; j < N - 1; j++) {
            for (let i = 1; i < N - 1; i++) {
                velocX[IX(i, j, N)] -= 0.5 * (p[IX(i + 1, j, N)] - p[IX(i - 1, j, N)]) * N;
                velocY[IX(i, j, N)] -= 0.5 * (p[IX(i, j + 1, N)] - p[IX(i, j - 1, N)]) * N;
            }
        }
        set_bnd(1, velocX, N);
        set_bnd(2, velocY, N);
    }

    function advect(b, d, d0, velocX, velocY, dt, N) {
        let i0, i1, j0, j1;
        let x, y, s0, t0, s1, t1;
        let dt0 = dt * (N - 2);
        let i, j;

        for (let j = 1; j < N - 1; j++) {
            for (let i = 1; i < N - 1; i++) {
                x = i - dt0 * velocX[IX(i, j, N)];
                y = j - dt0 * velocY[IX(i, j, N)];

                if (x < 0.5) x = 0.5;
                if (x > N + 0.5) x = N + 0.5;
                i0 = Math.floor(x);
                i1 = i0 + 1;

                if (y < 0.5) y = 0.5;
                if (y > N + 0.5) y = N + 0.5;
                j0 = Math.floor(y);
                j1 = j0 + 1;

                s1 = x - i0;
                s0 = 1.0 - s1;
                t1 = y - j0;
                t0 = 1.0 - t1;

                d[IX(i, j, N)] =
                    s0 * (t0 * d0[IX(i0, j0, N)] + t1 * d0[IX(i0, j1, N)]) +
                    s1 * (t0 * d0[IX(i1, j0, N)] + t1 * d0[IX(i1, j1, N)]);
            }
        }
        set_bnd(b, d, N);
    }

    function set_bnd(b, x, N) {
        for (let i = 1; i < N - 1; i++) {
            x[IX(i, 0, N)] = b === 2 ? -x[IX(i, 1, N)] : x[IX(i, 1, N)];
            x[IX(i, N - 1, N)] = b === 2 ? -x[IX(i, N - 2, N)] : x[IX(i, N - 2, N)];
        }
        for (let j = 1; j < N - 1; j++) {
            x[IX(0, j, N)] = b === 1 ? -x[IX(1, j, N)] : x[IX(1, j, N)];
            x[IX(N - 1, j, N)] = b === 1 ? -x[IX(N - 2, j, N)] : x[IX(N - 2, j, N)];
        }

        x[IX(0, 0, N)] = 0.5 * (x[IX(1, 0, N)] + x[IX(0, 1, N)]);
        x[IX(0, N - 1, N)] = 0.5 * (x[IX(1, N - 1, N)] + x[IX(0, N - 2, N)]);
        x[IX(N - 1, 0, N)] = 0.5 * (x[IX(N - 2, 0, N)] + x[IX(N - 1, 1, N)]);
        x[IX(N - 1, N - 1, N)] = 0.5 * (x[IX(N - 2, N - 1, N)] + x[IX(N - 1, N - 2, N)]);
    }

    class Particle {
        constructor() {
            this.reset(true);
        }

        reset(randomizeAge = false) {
            this.x = Math.random() * width;
            this.y = Math.random() * height;
            this.vx = 0;
            this.vy = 0;
            this.life = 100 + Math.random() * 100;
            this.age = randomizeAge ? Math.random() * this.life : 0;
            this.history = [];
        }

        update(fluidGrid) {
            // Map position to grid coordinates
            const gridX = (this.x / width) * fluidGrid.size;
            const gridY = (this.y / height) * fluidGrid.size;

            // Get velocity from grid
            const v = fluidGrid.getVelocity(gridX, gridY);

            // Advect
            const speedScale = 50;
            this.vx = v.x * speedScale;
            this.vy = v.y * speedScale;

            this.x += this.vx;
            this.y += this.vy;

            // Wrap around
            if (this.x < 0) this.x = width;
            if (this.x > width) this.x = 0;
            if (this.y < 0) this.y = height;
            if (this.y > height) this.y = 0;

            // Store history for tail
            this.history.push({ x: this.x, y: this.y });
            if (this.history.length > 5) this.history.shift();

            // Age
            this.age++;
            if (this.age > this.life) {
                this.reset();
            }
        }

        draw(ctx) {
            if (this.history.length < 2) return;

            const speed = Math.sqrt(this.vx * this.vx + this.vy * this.vy);
            const alpha = Math.min(0.6, speed * 0.3); // Fade out if slow

            ctx.beginPath();
            ctx.moveTo(this.history[0].x, this.history[0].y);

            // Draw simple trail
            for (let i = 1; i < this.history.length; i++) {
                const dx = Math.abs(this.history[i].x - this.history[i - 1].x);
                const dy = Math.abs(this.history[i].y - this.history[i - 1].y);
                if (dx < 50 && dy < 50) { // Don't draw across screen wrap
                    ctx.lineTo(this.history[i].x, this.history[i].y);
                } else {
                    ctx.moveTo(this.history[i].x, this.history[i].y);
                }
            }

            ctx.strokeStyle = `rgba(212, 175, 55, ${alpha})`;
            ctx.lineWidth = 1;
            ctx.stroke();
        }
    }

    // Initialize
    const fluid = new FluidGrid(N, diffusion, viscosity, dt);

    function resize() {
        width = window.innerWidth;
        height = window.innerHeight;
        canvas.width = width;
        canvas.height = height;
        canvas.style.width = '100%';
        canvas.style.height = '100%';
    }
    window.addEventListener('resize', resize);
    resize();

    // Init Particles
    for (let i = 0; i < numParticles; i++) {
        particles.push(new Particle());
    }

    // Interaction
    let isDragging = false;
    let lastX = 0;
    let lastY = 0;

    function handleInput(x, y) {
        const gridX = Math.floor((x / width) * N);
        const gridY = Math.floor((y / height) * N);

        const dx = (x - lastX) * 5;
        const dy = (y - lastY) * 5;

        fluid.addForce(gridX, gridY, dx, dy);

        lastX = x;
        lastY = y;
    }

    window.addEventListener('mousedown', e => {
        isDragging = true;
        lastX = e.clientX;
        lastY = e.clientY;
        handleInput(e.clientX, e.clientY);
    });

    window.addEventListener('mousemove', e => {
        if (isDragging) {
            handleInput(e.clientX, e.clientY);
        }
    });

    window.addEventListener('mouseup', () => isDragging = false);

    // Touch
    window.addEventListener('touchstart', e => {
        isDragging = true;
        lastX = e.touches[0].clientX;
        lastY = e.touches[0].clientY;
    }, { passive: false });

    window.addEventListener('touchmove', e => {
        if (isDragging) {
            handleInput(e.touches[0].clientX, e.touches[0].clientY);
        }
    }, { passive: false });

    window.addEventListener('touchend', () => isDragging = false);

    // Auto-flow
    function autoFlow() {
        // Continuous gentle flow
        if (Math.random() < 0.2) {
            const x = Math.floor(Math.random() * N);
            const y = Math.floor(Math.random() * N);
            fluid.addForce(x, y, (Math.random() - 0.5) * 5, (Math.random() - 0.5) * 5);
        }
    }

    function animate() {
        ctx.fillStyle = '#0a192f';
        ctx.fillRect(0, 0, width, height);

        autoFlow();
        fluid.step();

        ctx.globalCompositeOperation = 'lighter';
        for (let p of particles) {
            p.update(fluid);
            p.draw(ctx);
        }
        ctx.globalCompositeOperation = 'source-over';

        requestAnimationFrame(animate);
    }

    animate();
}

function initImageFadeTransition() {
    const neuroVisual = document.getElementById('neuro-visual');
    if (!neuroVisual) return;

    const images = neuroVisual.querySelectorAll('.fade-image');
    if (images.length < 2) return;

    let currentIndex = 0;

    // Ensure first image is visible
    images[0].classList.add('active');

    setInterval(() => {
        const nextIndex = (currentIndex + 1) % images.length;

        // Add active class to next image (starts fading in)
        images[nextIndex].classList.add('active');

        // After a brief moment, remove active from current (starts fading out)
        // This creates a crossfade effect
        setTimeout(() => {
            images[currentIndex].classList.remove('active');
            currentIndex = nextIndex;
        }, 50); // Small delay to ensure both transitions happen

    }, 5000); // Switch every 5 seconds
}
