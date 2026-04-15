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

struct Arena {
    arena: Vec<QuadNode>,
    index: usize,
}

impl Arena {
    fn new(capacity: usize) -> Self {
        let arena = Vec::with_capacity(capacity);
        let index = 0;
        Self { arena, index }
    }

    fn push(&mut self, node: QuadNode) -> usize {
        self.arena.push(node);
        self.index += 1;
        self.index
    }

    fn get_mut(&mut self, index: usize) -> &mut QuadNode {
        unsafe { self.arena.get_unchecked_mut(index) }
    }

    fn get(&self, index: usize) -> &QuadNode {
        unsafe { self.arena.get_unchecked(index) }
    }

    fn reset(&mut self) {
        self.arena.clear();
        self.index = 0;
    }
}

// An enum for quad tree, either holds a final node with Some(index of particle) or None
// or Branch, which is 4 quadrants within it
#[derive(Clone)]
enum QuadTree {
    Leaf(Option<usize>),
    Branch([usize; 4]),
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

impl QuadNode {
    fn with_width_height(width: f32, height: f32) -> Self {
        Self {
            top_left: Vec2::ZERO,
            width,
            height,
            center_of_mass_sum: Vec2::ZERO,
            mass: 0.,
            qt: QuadTree::Leaf(None),
        }
    }

    fn calculate_field(&self, i: usize, particles: &mut Vec<Particle>, arena: &Arena) {
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
                        let node = arena.get(branch[ix]);
                        node.calculate_field(i, particles, arena);
                    }
                }
            }
            _ => {}
        }
    }
}

fn put(
    parent_node_index: usize,
    particle_index: usize,
    particles: &Vec<Particle>,
    arena: &mut Arena,
) {
    let current_particle = particles[particle_index];
    let node = arena.get(parent_node_index);
    match &node.qt {
        QuadTree::Leaf(particle_container) => {
            // NLL magic
            let top_left = node.top_left;
            let height = node.height;
            let width = node.width;
            match *particle_container {
                // If the leaf is empty, simply add particle
                None => {
                    let node = arena.get_mut(parent_node_index);
                    node.qt = QuadTree::Leaf(Some(particle_index));
                }
                // If the leaf is occupied, split it into quadrants and to individual quad
                // The quadtree associated with the node becomes a branch
                Some(other_particle_index) => {
                    let base = QuadNode::with_width_height(node.width * 0.5, node.height * 0.5);
                    let i = arena.push(QuadNode {
                        top_left: Vec2 {
                            x: top_left.x + width / 2.,
                            y: top_left.y,
                        },
                        ..base.clone()
                    });
                    let j = arena.push(QuadNode {
                        top_left: Vec2 {
                            x: top_left.x,
                            y: top_left.y,
                        },
                        ..base.clone()
                    });
                    let k = arena.push(QuadNode {
                        top_left: Vec2 {
                            x: top_left.x,
                            y: top_left.y + height / 2.,
                        },
                        ..base.clone()
                    });
                    let l = arena.push(QuadNode {
                        top_left: Vec2 {
                            x: top_left.x + width / 2.,
                            y: top_left.y + height / 2.,
                        },
                        ..base
                    });

                    let node = arena.get_mut(parent_node_index);
                    node.qt = QuadTree::Branch([i, j, k, l]);
                    node.center_of_mass_sum += current_particle.mass * current_particle.r;
                    node.mass += current_particle.mass;
                    put(parent_node_index, other_particle_index, particles, arena);
                    put(parent_node_index, particle_index, particles, arena);
                }
            }
        }
        // check which leaf in branch has the coordinates required to fit in the particle, and try
        // putting in it
        QuadTree::Branch(branch) => {
            for ix in 0..4 {
                let branch_node = arena.get(branch[ix]);
                if branch_node.top_left.x < current_particle.r.x
                    && current_particle.r.x < branch_node.top_left.x + branch_node.width
                    && branch_node.top_left.y < current_particle.r.y
                    && current_particle.r.y < branch_node.top_left.y + branch_node.height
                {
                    put(branch[ix], particle_index, particles, arena);
                    break;
                }
            }
        }
    };
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

    let mut arena = Arena::new(N * 4);

    // Used for deltatime later on
    let dt = 1. / 60.;

    let mut stop = 0;
    loop {
        arena.reset();

        let root = QuadNode {
            top_left: Vec2::ZERO,
            width: WIDTH as f32,
            height: HEIGHT as f32,
            center_of_mass_sum: Vec2::ZERO,
            mass: 0.,
            qt: QuadTree::Leaf(None),
        };

        arena.arena.push(root);

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
            put(0, i, &particles, &mut arena);
        }

        for i in 0..particles.len() {
            let root = arena.get(0);
            root.calculate_field(i, &mut particles, &arena);
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
