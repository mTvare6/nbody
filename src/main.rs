use glam::f32::Vec2;
use rand::distributions::{Distribution, Uniform};

const WIDTH: u32 = 512;
const HEIGHT: u32 = 512;
const N: usize = 5_000;
const GRAV: f32 = 1.; // gravitation constant
const THETA: f32 = 0.4; // Parameter affected both quality and speed. Too high, quality is low, too low fps is low
const SOFTENING: f32 = 8.; // softening parameter based on wikipedia article on nbody
const M: f32 = 1e4; // mass of central body

#[derive(Clone, Copy)]
struct Particle {
    mass: f32,
    r: Vec2,
    v: Vec2,
    field: Vec2,
    field_prev: Vec2,
    first_iter: bool,
}

impl Default for Particle {
    fn default() -> Self {
        Self {
            mass: 1.,
            r: Vec2::ZERO,
            v: Vec2::ZERO,
            field: Vec2::ZERO,
            field_prev: Vec2::ZERO,
            first_iter: true,
        }
    }
}

// An enum for quad tree, either holds a final node with Some(index of particle) or None
// or Branch, which is 4 quadrants within it
#[derive(Clone)]
enum QuadTree {
    Leaf(Option<usize>),
    Branch([Box<QuadNode>; 4]),
}

// Metadata associated with Quadtree, such as position, width, height, mass contained within the
// node, total_mass*center_of_mass and quadtree itself
#[derive(Clone)]
struct QuadNode {
    top_left: Vec2,
    width: f32,
    height: f32,
    mass: f32,
    center_of_mass_sum: Vec2,
    qt: QuadTree,
}

impl Default for QuadNode {
    fn default() -> Self {
        Self {
            top_left: Vec2::ZERO,
            width: 0.,
            height: 0.,
            center_of_mass_sum: Vec2::ZERO,
            mass: 0.,
            qt: QuadTree::Leaf(None),
        }
    }
}

impl QuadNode {
    fn put(&mut self, i: usize, particles: &Vec<Particle>) {
        let p = particles[i];
        match &mut self.qt {
            QuadTree::Leaf(particle) => {
                self.center_of_mass_sum += p.mass * p.r;
                self.mass += p.mass;
                match particle.take() {
                    // If the leaf is empty, simply add particle
                    None => {
                        self.qt = QuadTree::Leaf(Some(i));
                    }
                    // If the leaf is occupied, split it into quadrants and to individual quad
                    // The quadtree associated with the node becomes a branch
                    Some(u) => {
                        let base = QuadNode {
                            width: self.width / 2.,
                            height: self.height / 2.,
                            ..Default::default()
                        };
                        self.qt = QuadTree::Branch([
                            Box::new(QuadNode {
                                top_left: Vec2 {
                                    x: self.top_left.x + self.width / 2.,
                                    y: self.top_left.y,
                                },
                                ..base.clone()
                            }),
                            Box::new(QuadNode {
                                top_left: Vec2 {
                                    x: self.top_left.x,
                                    y: self.top_left.y,
                                },
                                ..base.clone()
                            }),
                            Box::new(QuadNode {
                                top_left: Vec2 {
                                    x: self.top_left.x,
                                    y: self.top_left.y + self.height / 2.,
                                },
                                ..base.clone()
                            }),
                            Box::new(QuadNode {
                                top_left: Vec2 {
                                    x: self.top_left.x + self.width / 2.,
                                    y: self.top_left.y + self.height / 2.,
                                },
                                ..base
                            }),
                        ]);
                        self.put(u, particles);
                        self.put(i, particles);
                    }
                }
            }
            // check which leaf in branch has the coordinates required to fit in the particle, and try
            // putting in it
            QuadTree::Branch(branch) => {
                for ix in 0..4 {
                    if branch[ix].top_left.x < p.r.x
                        && p.r.x < branch[ix].top_left.x + branch[ix].width
                        && branch[ix].top_left.y < p.r.y
                        && p.r.y < branch[ix].top_left.y + branch[ix].height
                    {
                        branch[ix].put(i, particles);
                        break;
                    }
                }
            }
        };
    }

    fn calculate_field(&self, i: usize, particles: &mut Vec<Particle>) {
        match &self.qt {
            // for a node with single particle, ie leaf, calculate the distance separate
            &QuadTree::Leaf(Some(j)) => {
                if i != j {
                    let r = particles[j].r - particles[i].r;
                    let d = (r.length_squared() + SOFTENING * SOFTENING).sqrt();
                    let m = particles[j].mass;
                    let scale = GRAV * m / (d * d * d);
                    let field = scale * r;
                    particles[i].field += field;
                }
            }
            // if its a branch, either check if the condition for approximation holds true, or recursve
            // through the tree until all particles are included
            QuadTree::Branch(branch) => {
                let m = self.mass;
                let r = self.center_of_mass_sum / m - particles[i].r;
                let d = (r.length_squared() + SOFTENING * SOFTENING).sqrt();
                let s = self.width;
                if s / d <= THETA {
                    let scale = GRAV * m / (d * d * d);
                    particles[i].field += r * scale;
                } else {
                    for ix in 0..4 {
                        branch[ix].calculate_field(i, particles);
                    }
                }
            }
            _ => {}
        }
    }
}

fn main() {
    let mut particles: Vec<Particle> = Vec::new();
    let mut rng = rand::thread_rng();

    let uradius = Uniform::from(30.0..100.);
    let utheta = Uniform::from(-std::f32::consts::PI..std::f32::consts::PI);

    particles.push(Particle {
        mass: M,
        r: Vec2 {
            x: (WIDTH / 2) as f32,
            y: (HEIGHT / 2) as f32,
        },
        ..Default::default()
    });
    // Make N random particles spread around center
    for _ in 1..=N {
        let r = uradius.sample(&mut rng);
        let t = utheta.sample(&mut rng);
        let v = (GRAV * M / r).sqrt();
        particles.push(Particle {
            mass: 1.,
            r: Vec2 {
                x: (WIDTH / 2) as f32 + r * t.cos(),
                y: (HEIGHT / 2) as f32 + r * t.sin(),
            },
            v: Vec2 {
                x: -v * t.sin(),
                y: v * t.cos(),
            },
            // r is perpendicular to v
            ..Default::default()
        });
    }

    // let mut arena: Vec<QuadNode> = Vec::with_capacity(N * 4);

    // Used for deltatime later on
    let dt = 1. / 60.;

    let mut stop = 0;
    loop {
        let mut root = QuadNode {
            top_left: Vec2::ZERO,
            width: WIDTH as f32,
            height: HEIGHT as f32,
            center_of_mass_sum: Vec2::ZERO,
            mass: 0.,
            qt: QuadTree::Leaf(None),
        };

        // build tree
        for i in 0..particles.len() {
            // If particle is not within the screen, respawm it as a new particle somewhere inside
            if !(0. < particles[i].r.x
                && particles[i].r.x < WIDTH as f32
                && 0. < particles[i].r.y
                && particles[i].r.y < HEIGHT as f32)
            {
                let r = uradius.sample(&mut rng);
                let t = utheta.sample(&mut rng);
                particles[i].r = Vec2 {
                    x: (WIDTH / 2) as f32 + r * t.cos(),
                    y: (HEIGHT / 2) as f32 + r * t.sin(),
                };
                particles[i].v = Vec2 {
                    x: -r / 100. * t.sin(),
                    y: r / 100. * t.cos(),
                };
                particles[i].field = Vec2::ZERO;
                particles[i].first_iter = true;
            }
            // Try adding the particle to root
            root.put(i, &particles);
        }

        for i in 0..particles.len() {
            root.calculate_field(i, &mut particles);
        }
        for i in 0..particles.len() {
            // using particles[i].v = particles[i].field*dt returns an error, telling to use a
            // local variable, so instead handle each field
            if !particles[i].first_iter {
                particles[i].v.x += dt / 2. * (particles[i].field_prev.x + particles[i].field.x);
                particles[i].v.y += dt / 2. * (particles[i].field_prev.y + particles[i].field.y);
            } else {
                particles[i].first_iter = false;
            }
            // TODO: Decide if it is efficient to have a bool check everytime despite the
            // fact only checking once is needed. Otherwise make a separate first run outside the loop just
            // for this(but then code duplication)
            particles[i].field_prev = particles[i].field;
            particles[i].field = Vec2::ZERO;
            particles[i].r.x += particles[i].v.x * dt + particles[i].field.x / 2. * dt * dt;
            particles[i].r.y += particles[i].v.y * dt + particles[i].field.y / 2. * dt * dt;
        }

        stop += 1;

        if stop == 1000 {
            break;
        }
    }
}
