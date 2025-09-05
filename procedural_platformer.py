#
"""


Features
- Infinite scrolling world driven by deterministic 1D gradient noise (Perlin-like).
- Smooth rolling ground surface (a function, not tiles) with slopes and hills.
- On-the-fly generation of floating platforms, collectibles, and hazards in "chunks".
- Simple, responsive platforming controls (left/right, jump, coyote time, variable jump).
- Parallax sky layers, day-night tinting, and camera follow.
- Basic HUD and pause/restart.
- Deterministic world via a visible seed.

Dependencies
- Python 3.8+
- pygame (pip install pygame)

Controls
- Left/Right or A/D: move
- Space / W / Up: jump (with coyote time and jump buffering)
- Esc: pause
- R: restart at current seed
- N: new seed (re-roll world) while paused
- F1: show/hide debug overlay

Author: Donfeur / Balazs
"""
import math
import random
import sys
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import pygame

# ----------------------------- Config ---------------------------------
WIDTH, HEIGHT = 960, 540
SCALE = 1
FPS = 165

GRAVITY = 2000.0            # px/s^2
MOVE_SPEED = 320.0          # px/s
AIR_ACCEL = 2200.0          # px/s^2
GROUND_ACCEL = 5000.0       # px/s^2
FRICTION = 0.86             # horizontal damping when no input on ground
JUMP_VELOCITY = 760.0       # px/s
JUMP_CUTOFF = 0.45          # release jump to cut to this fraction of v
COYOTE_TIME = 0.10          # seconds after leaving ground that you can still jump
JUMP_BUFFER = 0.12          # seconds before landing to buffer a jump

PLAYER_W, PLAYER_H = 36, 52

# World
CHUNK_W = 800               # pixels per procedural content chunk
GROUND_BASE = HEIGHT * 0.70
GROUND_AMP = 140.0          # hill amplitude
HAZARD_Y_OFFSET = 18        # spike height above ground

# Platforms
PLATFORM_MIN_Y = int(HEIGHT * 0.25)
PLATFORM_MAX_Y = int(HEIGHT * 0.62)
PLATFORM_RATE = 0.6         # per chunk
PLATFORM_W_RANGE = (90, 190)

# Collectibles & hazards
GEM_RATE = 1.6              # per chunk
SPIKE_RATE = 0.9            # per chunk

# Rendering
SKY_COLOR_TOP = (32, 54, 94)
SKY_COLOR_BOTTOM = (160, 195, 255)
GROUND_COLOR = (78, 120, 73)
DIRT_COLOR = (63, 90, 60)
PLATFORM_COLOR = (100, 100, 110)
GEM_COLOR = (255, 210, 60)
SPIKE_COLOR = (200, 60, 60)
PLAYER_COLOR = (240, 240, 255)

PARALLAX_LAYERS = [
    # (speed multiplier, alpha, density per 1000px, height range)
    (0.2, 90, 0.7, (HEIGHT*0.05, HEIGHT*0.35)),
    (0.4, 120, 1.1, (HEIGHT*0.10, HEIGHT*0.50)),
    (0.7, 160, 1.7, (HEIGHT*0.25, HEIGHT*0.65)),
]

# ----------------------------- Utilities ------------------------------
def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def lerp(a, b, t):
    return a + (b - a) * t

def smoothstep(t):
    # 6t^5 - 15t^4 + 10t^3
    return t * t * t * (t * (t * 6 - 15) + 10)

# ----------------------------- 1D Gradient Noise ----------------------
class Noise1D:
    """Deterministic Perlin-like 1D gradient noise with octaves."""
    def __init__(self, seed: int):
        self.seed = seed

    def _hash_grad(self, ix: int) -> float:
        # Pseudo-random gradient in [-1, 1] from integer coordinate
        n = (ix * 374761393 + self.seed * 668265263) & 0xFFFFFFFF
        n = (n ^ (n >> 13)) * 1274126177 & 0xFFFFFFFF
        n = (n ^ (n >> 16)) & 0xFFFFFFFF
        # Map to [-1, 1]
        return ((n / 0xFFFFFFFF) * 2.0) - 1.0

    def value(self, x: float, frequency: float = 1.0, octaves: int = 4, lacunarity: float = 2.0, gain: float = 0.5) -> float:
        """Return smooth noise in [-1, 1] at float x."""
        amp = 1.0
        freq = frequency
        total = 0.0
        norm = 0.0
        for _ in range(octaves):
            # Gradient noise: pick gradients at floor(x), floor(x)+1
            xf = x * freq
            xi = math.floor(xf)
            frac = xf - xi

            g0 = self._hash_grad(xi)
            g1 = self._hash_grad(xi + 1)

            v0 = g0 * frac            # dot((frac, 0), (g0, 0)) in 1D
            v1 = g1 * (frac - 1.0)

            t = smoothstep(frac)
            n = lerp(v0, v1, t) * 2.0  # scale a bit

            total += n * amp
            norm += amp
            amp *= gain
            freq *= lacunarity
        return total / max(1e-6, norm)

# ----------------------------- World Types ----------------------------
@dataclass
class Platform:
    rect: pygame.Rect

@dataclass
class Spike:
    pos: Tuple[int, int]  # tip position

@dataclass
class Gem:
    pos: Tuple[int, int]
    taken: bool = False

@dataclass
class Chunk:
    index: int
    platforms: List[Platform] = field(default_factory=list)
    spikes: List[Spike] = field(default_factory=list)
    gems: List[Gem] = field(default_factory=list)

# ----------------------------- World Generator ------------------------
class World:
    def __init__(self, seed: int):
        self.seed = seed
        self.noise = Noise1D(seed)
        self.chunks = {}  # index -> Chunk
        self.rng = random.Random(seed)

    def ground_height(self, x: float) -> float:
        # Smooth rolling ground
        n = self.noise.value(x * 0.0023, frequency=1.0, octaves=5, lacunarity=2.0, gain=0.55)
        m = self.noise.value(x * 0.0006 + 1000.0, frequency=1.0, octaves=3, lacunarity=2.0, gain=0.6)
        h = GROUND_BASE + GROUND_AMP * (0.6 * n + 0.4 * m)
        return clamp(h, HEIGHT * 0.35, HEIGHT * 0.90)

    def _chunk_rng(self, index: int) -> random.Random:
        return random.Random((self.seed << 20) ^ (index * 2654435761))

    def ensure_chunk(self, index: int):
        if index in self.chunks:
            return
        rng = self._chunk_rng(index)
        chunk = Chunk(index)

        # Platforms
        if rng.random() < PLATFORM_RATE:
            num = 1 + int(rng.random() * 2)
            for _ in range(num):
                w = rng.randint(*PLATFORM_W_RANGE)
                px = index * CHUNK_W + rng.randint(40, CHUNK_W - 60 - w)
                py = rng.randint(PLATFORM_MIN_Y, PLATFORM_MAX_Y)
                # Skip if inside ground
                if py + 12 > self.ground_height(px):
                    py = int(self.ground_height(px) - rng.randint(50, 120))
                rect = pygame.Rect(px, py, w, 16)
                chunk.platforms.append(Platform(rect))

        # Spikes near ground
        if rng.random() < SPIKE_RATE:
            count = 1 + rng.randint(0, 2)
            for _ in range(count):
                sx = index * CHUNK_W + rng.randint(60, CHUNK_W - 40)
                sy = int(self.ground_height(sx) - HAZARD_Y_OFFSET)
                chunk.spikes.append(Spike((sx, sy)))

        # Gems
        if rng.random() < GEM_RATE:
            count = 2 + rng.randint(0, 4)
            for _ in range(count):
                gx = index * CHUNK_W + rng.randint(40, CHUNK_W - 40)
                gy = int(self.ground_height(gx) - rng.randint(90, 180))
                # Avoid too high
                gy = max(HEIGHT * 0.10, min(gy, HEIGHT * 0.70))
                chunk.gems.append(Gem((gx, int(gy))))

        self.chunks[index] = chunk

    def visible_chunks(self, cam_x: float) -> List[Chunk]:
        start_idx = int((cam_x - WIDTH * 0.5) // CHUNK_W) - 1
        end_idx = int((cam_x + WIDTH * 1.5) // CHUNK_W) + 1
        out = []
        for idx in range(start_idx, end_idx + 1):
            self.ensure_chunk(idx)
            out.append(self.chunks[idx])
        return out

# ----------------------------- Player ---------------------------------
class Player:
    def __init__(self, x: float, y: float):
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(0, 0)
        self.on_ground = False
        self.coyote = 0.0
        self.jump_buf = 0.0
        self.rect = pygame.Rect(int(x), int(y), PLAYER_W, PLAYER_H)
        self.facing = 1

    def update_rect(self):
        self.rect.topleft = (int(self.pos.x), int(self.pos.y))


# ----------------------------- Projectiles -----------------------------
class Bullet:
    def __init__(self, x, y, dx, dy):
        self.pos = pygame.Vector2(x, y)
        # Normalize direction and apply speed
        if dx == 0 and dy == 0:
            dx, dy = 1, 0
        self.vel = pygame.Vector2(dx, dy).normalize() * 700  # px/s
        self.life = 1.6  # seconds alive

    def update(self, dt):
        self.pos += self.vel * dt
        self.life -= dt
        return self.life > 0  # return False when bullet expires

    def draw(self, screen, to_screen):
        x, y = to_screen(self.pos.x, self.pos.y)
        pygame.draw.circle(screen, (250, 220, 60), (x, y), 5)

# ----------------------------- Game -----------------------------------
class Game:
    def __init__(self, seed: int = None):
        if seed is None:
            seed = random.randint(1, 1_000_000_000)
        self.seed = seed
        pygame.init()
        pygame.display.set_caption("Procedural Platformer")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16)
        self.big_font = pygame.font.SysFont("consolas", 24, bold=True)

        self.state = "menu"   # can be "menu" or "playing"
    def draw_menu(self):
        self.screen.fill((30, 30, 60))
        title = self.big_font.render("Procedural Platformer", True, (255, 255, 255))
        prompt = self.font.render("Press ENTER to Play, ESC to Quit", True, (200, 200, 200))
        
        # Center text
        self.screen.blit(title, (WIDTH // 2 - title.get_width() // 2, HEIGHT // 3))
        self.screen.blit(prompt, (WIDTH // 2 - prompt.get_width() // 2, HEIGHT // 2))
        
        pygame.display.flip()


        # --- SOUND SETUP ---
        pygame.mixer.init()

        # Load short sound effects
        self.snd_jump = pygame.mixer.Sound("assets/jump.wav")
        self.snd_shoot = pygame.mixer.Sound("assets/shoot.wav")
        self.snd_gem = pygame.mixer.Sound("assets/gem.wav")
        self.snd_die = pygame.mixer.Sound("assets/die.wav")

        # (optional) Background music
        # pygame.mixer.music.load("assets/music.ogg")
        # pygame.mixer.music.set_volume(0.4)
        # pygame.mixer.music.play(-1)   # loop forever

      #  self.seed = seed if seed is not None else random.randint(1, 1_000_000_000)
        self.reset_world(self.seed)

        self.pause = False
        self.show_debug = False

        # NEW: hide OS cursor & setup bullet storage
        pygame.mouse.set_visible(False)
        self.bullets = []


    def reset_world(self, seed: int):
        self.world = World(seed)
        # place player slightly above ground at x=0
        start_x = 60
        ground_y = self.world.ground_height(start_x)
        self.player = Player(start_x, ground_y - PLAYER_H - 8)
        self.cam_x = 0.0
        self.cam_y = 0.0
        self.score = 0
        self.deaths = 0
        self.time_start = time.time()

    # ------------------------- Physics Helpers ------------------------
    def _ground_y_span(self, x0: float, x1: float, samples: int = 4) -> float:
        # Return minimum ground height across [x0, x1] for safety in slopes
        s = max(2, samples)
        hmin = 1e9
        for i in range(s):
            t = i / (s - 1)
            x = lerp(x0, x1, t)
            h = self.world.ground_height(x)
            if h < hmin:
                hmin = h
        return hmin

    def _move_and_collide(self, dt: float):
        p = self.player
        p.on_ground = False

        # Horizontal move
        new_x = p.pos.x + p.vel.x * dt
        # Collide with platforms horizontally
        test_rect = pygame.Rect(int(new_x), int(p.pos.y), PLAYER_W, PLAYER_H)
        colliders = self._nearby_platform_rects()
        hit_horiz = None
        for r in colliders:
            if test_rect.colliderect(r):
                if p.vel.x > 0:
                    new_x = r.left - PLAYER_W
                elif p.vel.x < 0:
                    new_x = r.right
                p.vel.x = 0
                hit_horiz = r
                break
        p.pos.x = new_x

        # Vertical move (integrate)
        new_y = p.pos.y + p.vel.y * dt
        test_rect = pygame.Rect(int(p.pos.x), int(new_y), PLAYER_W, PLAYER_H)
        hit_vert = None
        for r in colliders:
            if test_rect.colliderect(r):
                if p.vel.y > 0:
                    new_y = r.top - PLAYER_H
                    p.on_ground = True
                    p.coyote = COYOTE_TIME
                elif p.vel.y < 0:
                    new_y = r.bottom
                p.vel.y = 0
                hit_vert = r
                break
        p.pos.y = new_y

        # Ground collision with height field
        feet_left = p.pos.x + 6
        feet_right = p.pos.x + PLAYER_W - 6
        ground_y = self._ground_y_span(feet_left, feet_right, samples=6)
        if p.pos.y + PLAYER_H >= ground_y:
            if p.vel.y >= 0:  # falling or resting
                p.pos.y = ground_y - PLAYER_H
                p.vel.y = 0
                p.on_ground = True
                p.coyote = COYOTE_TIME

        # Spikes
        for spike in self._nearby_spikes():
            sx, sy = spike.pos
            # simple triangle collision: if feet below spike tip and horizontally close
            if feet_left <= sx <= feet_right and p.pos.y + PLAYER_H >= sy - 4:
                self._die()
                break

        # Gems
        for gem in self._nearby_gems():
            if not gem.taken:
                gx, gy = gem.pos
                if pygame.Rect(gx - 10, gy - 10, 20, 20).colliderect(p.rect):
                    gem.taken = True
                    self.score += 10
                    self.snd_gem.play()

        p.update_rect()

    def _die(self):
        self.snd_die.play()
        self.deaths += 1
        # Respawn slightly back
        respawn_x = max(20, self.player.pos.x - 250)
        ground_y = self.world.ground_height(respawn_x)
        self.player.pos.update(respawn_x, ground_y - PLAYER_H - 8)
        self.player.vel.update(0, 0)
        self.cam_shake_t = 0.25

    # --------------------------- Queries ------------------------------
    def _nearby_platform_rects(self) -> List[pygame.Rect]:
        rects = []
        for ch in self.world.visible_chunks(self.cam_x):
            for p in ch.platforms:
                rects.append(p.rect)
        return rects

    def _nearby_spikes(self) -> List[Spike]:
        spikes = []
        for ch in self.world.visible_chunks(self.cam_x):
            spikes.extend(ch.spikes)
        return spikes

    def _nearby_gems(self) -> List[Gem]:
        gems = []
        for ch in self.world.visible_chunks(self.cam_x):
            gems.extend(ch.gems)
        return gems

    # --------------------------- Update --------------------------------
    def handle_input(self, dt: float):
        p = self.player
        keys = pygame.key.get_pressed()

        target = 0.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            target -= MOVE_SPEED
            p.facing = -1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            target += MOVE_SPEED
            p.facing = 1

        accel = GROUND_ACCEL if p.on_ground else AIR_ACCEL
        # accelerate toward target speed
        if target != 0:
            if p.vel.x < target:
                p.vel.x = min(target, p.vel.x + accel * dt)
            elif p.vel.x > target:
                p.vel.x = max(target, p.vel.x - accel * dt)
        else:
            if p.on_ground:
                p.vel.x *= FRICTION
                if abs(p.vel.x) < 6:
                    p.vel.x = 0.0

        # jumping
        want_jump = keys[pygame.K_SPACE] or keys[pygame.K_w] or keys[pygame.K_UP]
        if want_jump:
            self.player.jump_buf = JUMP_BUFFER

        # coyote / buffer
        if (p.on_ground or p.coyote > 0) and p.jump_buf > 0:
            p.vel.y = -JUMP_VELOCITY
            p.on_ground = False
            p.coyote = 0.0
            p.jump_buf = 0.0
            self.snd_jump.play()

        # variable jump height (cut velocity on release)
        if not want_jump and p.vel.y < -JUMP_CUTOFF * JUMP_VELOCITY:
            p.vel.y = -JUMP_CUTOFF * JUMP_VELOCITY

        # timers
        p.coyote = max(0.0, p.coyote - dt)
        p.jump_buf = max(0.0, p.jump_buf - dt)

    def update(self, dt: float):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.pause = not self.pause
                if event.key == pygame.K_F1:
                    self.show_debug = not self.show_debug
                if self.pause and event.key == pygame.K_r:
                    self.reset_world(self.seed)
                if self.pause and event.key == pygame.K_n:
                    self.seed = random.randint(1, 1_000_000_000)
                    self.reset_world(self.seed)
            
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Fire bullet toward cursor
                mx, my = pygame.mouse.get_pos()
                wx = self.cam_x - WIDTH // 2 + mx
                wy = self.cam_y - HEIGHT // 2 + my
                px = self.player.pos.x + PLAYER_W // 2
                py = self.player.pos.y + PLAYER_H // 2
                dx, dy = wx - px, wy - py
                self.bullets.append(Bullet(px, py, dx, dy))
                self.snd_shoot.play()


        if self.pause:
            return

        dt = clamp(dt, 0.0, 1.0 / 20.0)  # avoid huge steps

        self.handle_input(dt)

        # gravity
        self.player.vel.y += GRAVITY * dt
        self._move_and_collide(dt)
     # update bullets
        self.bullets = [b for b in self.bullets if b.update(dt)]


        # camera follow
        target_cam_x = self.player.pos.x + PLAYER_W * 0.5 + 160
        self.cam_x = lerp(self.cam_x, target_cam_x, 0.12)
        target_cam_y = self.player.pos.y - 120
        self.cam_y = lerp(self.cam_y, target_cam_y, 0.08)

    # --------------------------- Render --------------------------------
    def _world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        return int(x - self.cam_x + WIDTH // 2), int(y - self.cam_y + HEIGHT // 2)

    def draw_background(self):
        # vertical gradient sky
        top = pygame.Color(*SKY_COLOR_TOP)
        bottom = pygame.Color(*SKY_COLOR_BOTTOM)
        for y in range(HEIGHT):
            t = y / (HEIGHT - 1)
            c = top.lerp(bottom, t)
            pygame.draw.line(self.screen, c, (0, y), (WIDTH, y))

        # parallax hills (noise-based silhouettes)
        for speed, alpha, density, (ymin, ymax) in PARALLAX_LAYERS:
            surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            surf.set_alpha(alpha)
            # draw points
            step = 14
            pts = []
            x0_world = self.cam_x - WIDTH // 2
            for sx in range(0, WIDTH + step, step):
                wx = x0_world + sx
                h = self.world.noise.value(wx * 0.001 * speed + 2000, octaves=3)  # silhouette noise
                yy = lerp(ymin, ymax, (h + 1) * 0.5)
                pts.append((sx, int(yy)))
            # close polygon
            pts.append((WIDTH, HEIGHT))
            pts.append((0, HEIGHT))
            pygame.draw.polygon(surf, (40, 60, 80), pts)
            self.screen.blit(surf, (0, 0))

    def draw_ground(self):
        # Draw ground as thick polygon under the sampled curve
        step = 6
        pts_top = []
        x0_world = self.cam_x - WIDTH // 2 - 30
        for sx in range(-30, WIDTH + 60, step):
            wx = x0_world + sx
            gy = self.world.ground_height(wx)
            x, y = self._world_to_screen(wx, gy)
            pts_top.append((x, y))
        # close polygon to bottom
        pts = pts_top + [(WIDTH + 60, HEIGHT + 60), (-60, HEIGHT + 60)]
        pygame.draw.polygon(self.screen, GROUND_COLOR, pts)

        # dark edge
        pygame.draw.lines(self.screen, DIRT_COLOR, False, pts_top, 3)

    def draw_platforms(self):
        for ch in self.world.visible_chunks(self.cam_x):
            for pf in ch.platforms:
                x, y = self._world_to_screen(pf.rect.x, pf.rect.y)
                pygame.draw.rect(self.screen, PLATFORM_COLOR, pygame.Rect(x, y, pf.rect.w, pf.rect.h), border_radius=6)

    def draw_spikes(self):
        for ch in self.world.visible_chunks(self.cam_x):
            for sp in ch.spikes:
                sx, sy = self._world_to_screen(*sp.pos)
                size = 16
                pts = [(sx - size//2, sy), (sx + size//2, sy), (sx, sy - size)]
                pygame.draw.polygon(self.screen, SPIKE_COLOR, pts)

    def draw_gems(self):
        for ch in self.world.visible_chunks(self.cam_x):
            for gm in ch.gems:
                if gm.taken: 
                    continue
                x, y = self._world_to_screen(*gm.pos)
                r = 6
                pygame.draw.polygon(self.screen, GEM_COLOR, [(x, y - r), (x + r, y), (x, y + r), (x - r, y)])
                pygame.draw.circle(self.screen, (255, 255, 255), (x, y - r//2), 2)

    def draw_player(self):
        p = self.player
        x, y = self._world_to_screen(p.pos.x, p.pos.y)
        body = pygame.Rect(x, y, PLAYER_W, PLAYER_H)
        pygame.draw.rect(self.screen, PLAYER_COLOR, body, border_radius=8)
        # simple face
        eye_dx = 10 if p.facing >= 0 else PLAYER_W - 16
        pygame.draw.circle(self.screen, (20, 20, 40), (x + eye_dx, y + 16), 4)

    def draw_hud(self):
        t = time.time() - self.time_start
        hud = f"Seed: {self.world.seed}   Score: {self.score}   Deaths: {self.deaths}   Time: {t:5.1f}s   X: {int(self.player.pos.x)}"
        surf = self.font.render(hud, True, (15, 15, 20))
        self.screen.blit(surf, (10, 10))

        if self.pause:
            s = self.big_font.render("PAUSED  (R)estart  (N)ew seed  (Esc) Resume", True, (255, 255, 255))
            rect = s.get_rect(center=(WIDTH // 2, HEIGHT // 2))
            pygame.draw.rect(self.screen, (0, 0, 0, 150), rect.inflate(40, 20))
            self.screen.blit(s, rect)

        if self.show_debug:
            p = self.player
            lines = [
                f"pos=({p.pos.x:7.2f},{p.pos.y:7.2f}) vel=({p.vel.x:7.2f},{p.vel.y:7.2f})",
                f"ground_y={self.world.ground_height(p.pos.x):.2f} on_ground={p.on_ground} coyote={p.coyote:.2f}",
                f"chunks={sorted([c.index for c in self.world.visible_chunks(self.cam_x)])[:6]} ...",
            ]
            for i, txt in enumerate(lines):
                s = self.font.render(txt, True, (0, 0, 0))
                self.screen.blit(s, (10, 30 + 18 * i))

    def render(self):
        self.draw_background()
        self.draw_ground()
        self.draw_platforms()
        self.draw_spikes()
        self.draw_gems()
        self.draw_player()
        for b in self.bullets:
            b.draw(self.screen, self._world_to_screen)
        self.draw_cursor()
        self.draw_hud()

    def draw_cursor(self):
        mx, my = pygame.mouse.get_pos()
        # crosshair style cursor
        pygame.draw.circle(self.screen, (255, 80, 80), (mx, my), 6, 2)
        pygame.draw.line(self.screen, (255, 80, 80), (mx - 10, my), (mx + 10, my), 1)
        pygame.draw.line(self.screen, (255, 80, 80), (mx, my - 10), (mx, my + 10), 1)


    def run(self):
        running = True
        while running:
            self.clock.tick(120)

            if self.state == "menu":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RETURN:
                            self.init__(self.seed) # reinitialize the game
                            self.state = "playing"
                        elif event.key == pygame.K_ESCAPE:
                            running = False
                self.draw_menu()
                continue   # skip game logic when in menu



def main():
    seed = None
    if len(sys.argv) >= 2:
        try:
            seed = int(sys.argv[1])
        except Exception:
            seed = None
    g = Game(seed)
    g.run()


if __name__ == "__main__":
    main()
