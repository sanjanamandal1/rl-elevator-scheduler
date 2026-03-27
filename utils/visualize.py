import pygame
import numpy as np
import sys
import math
import config

# ── Colour palette ────────────────────────────────────────────────────────────
BG          = ( 15,  15,  26)
PANEL_BG    = ( 22,  22,  42)
SHAFT_BG    = ( 18,  18,  32)
FLOOR_LINE  = ( 38,  38,  58)
WHITE       = (224, 224, 255)
GRAY        = ( 85,  85, 112)
BLUE        = ( 55, 138, 221)
GREEN       = ( 29, 158, 117)
AMBER       = (239, 159,  39)
RED         = (226,  75,  74)
PURPLE      = (127, 119, 221)
TEAL        = ( 29, 158, 140)
SCAN_RED    = (200,  60,  60)

ELEV_COLORS   = [BLUE, GREEN, PURPLE, TEAL]
PRIORITY_COLS = {"normal": BLUE, "vip": AMBER, "emergency": RED}

# ── Layout constants ──────────────────────────────────────────────────────────
W, H         = 1200, 740
BLDG_X       = 20
BLDG_W       = 420
TOP_PAD      = 50
BOT_PAD      = 40
STATS_X      = BLDG_X + BLDG_W + 24
STATS_W      = W - STATS_X - 20


class ElevatorDashboard:
    def __init__(self, env, agent, agent_name="DQN", fps=10):
        pygame.init()
        pygame.display.set_caption(f"RL Elevator Scheduler — {agent_name}")

        self.env        = env
        self.agent      = agent
        self.agent_name = agent_name
        self.fps        = fps
        self.screen     = pygame.display.set_mode((W, H))
        self.clock      = pygame.time.Clock()

        # fonts
        self.f_tiny  = pygame.font.SysFont("Courier New", 10)
        self.f_sm    = pygame.font.SysFont("Courier New", 12)
        self.f_md    = pygame.font.SysFont("Courier New", 14, bold=True)
        self.f_lg    = pygame.font.SysFont("Courier New", 20, bold=True)

        # computed floor geometry
        avail          = H - TOP_PAD - BOT_PAD
        self.FLOOR_H   = avail // config.NUM_FLOORS
        self.BLDG_H    = self.FLOOR_H * config.NUM_FLOORS

        # shaft x positions (one per elevator, evenly spaced inside building panel)
        n              = config.NUM_ELEVATORS
        shaft_area_x   = BLDG_X + 60         # leave room for floor labels
        shaft_area_w   = BLDG_W - 70
        self.SHAFT_W   = min(32, shaft_area_w // n - 6)
        self.shaft_xs  = [
            shaft_area_x + i * (self.SHAFT_W + 6)
            for i in range(n)
        ]

        # tracking
        self.reward_hist  = []
        self.wait_hist    = []
        self.delivered    = 0
        self.step         = 0
        self.paused       = False
        self.scan_baseline = -810   # from your results — drawn as reference line

    # ── helpers ───────────────────────────────────────────────────────────────
    def _floor_y(self, floor):
        """Top-left y of a given floor row (floor 0 = bottom)."""
        return TOP_PAD + (config.NUM_FLOORS - 1 - floor) * self.FLOOR_H

    def _draw_text(self, surf, text, font, color, x, y, anchor="topleft"):
        s = font.render(text, True, color)
        r = s.get_rect(**{anchor: (x, y)})
        surf.blit(s, r)

    def _rounded_rect(self, surf, color, rect, radius=6, border=0, border_color=None):
        pygame.draw.rect(surf, color, rect, border_radius=radius)
        if border and border_color:
            pygame.draw.rect(surf, border_color, rect, border, border_radius=radius)

    # ── building panel ────────────────────────────────────────────────────────
    def _draw_building(self):
        # panel background
        self._rounded_rect(
            self.screen, PANEL_BG,
            (BLDG_X, TOP_PAD - 4, BLDG_W, self.BLDG_H + 8), radius=8
        )

        for f in range(config.NUM_FLOORS):
            y = self._floor_y(f)

            # floor separator line
            pygame.draw.line(self.screen, FLOOR_LINE,
                             (BLDG_X, y), (BLDG_X + BLDG_W, y), 1)

            # floor number label
            self._draw_text(self.screen, f"F{f+1:02d}", self.f_tiny,
                            GRAY, BLDG_X + 4, y + self.FLOOR_H // 2 - 5)

            # waiting passenger dots
            queue = self.env.building.waiting[f]
            dot_x = BLDG_X + 52
            for i, p in enumerate(queue[:14]):
                col = PRIORITY_COLS.get(getattr(p, "priority", "normal"), BLUE)
                cx  = dot_x + i * 9
                cy  = y + self.FLOOR_H // 2
                pygame.draw.circle(self.screen, col, (cx, cy), 3)

            # hall call indicator (amber square top-right of row)
            if queue:
                hc_rect = (BLDG_X + BLDG_W - 14, y + 3, 7, 7)
                pygame.draw.rect(self.screen, AMBER, hc_rect, border_radius=1)

    # ── elevator shafts & cars ────────────────────────────────────────────────
    def _draw_elevators(self):
        for i, elev in enumerate(self.env.elevators):
            sx    = self.shaft_xs[i]
            color = ELEV_COLORS[i % len(ELEV_COLORS)]

            # draw full shaft (faint)
            shaft_rect = (sx, TOP_PAD, self.SHAFT_W, self.BLDG_H)
            pygame.draw.rect(self.screen, SHAFT_BG, shaft_rect, border_radius=4)

            # elevator car
            cy     = self._floor_y(elev.current_floor)
            cab_h  = max(16, self.FLOOR_H - 3)
            cab    = (sx, cy + 2, self.SHAFT_W, cab_h)
            self._rounded_rect(self.screen, color, cab, radius=4)

            # passenger count inside car
            if elev.passengers:
                self._draw_text(
                    self.screen, str(len(elev.passengers)),
                    self.f_tiny, WHITE,
                    sx + self.SHAFT_W // 2, cy + cab_h // 2 - 4, anchor="center"
                )

            # direction arrow above/below car
            mx = sx + self.SHAFT_W // 2
            if elev.direction == 1:
                pts = [(mx, cy - 1), (mx - 5, cy + 7), (mx + 5, cy + 7)]
                pygame.draw.polygon(self.screen, WHITE, pts)
            elif elev.direction == -1:
                pts = [(mx, cy + cab_h + 1), (mx - 5, cy + cab_h - 7), (mx + 5, cy + cab_h - 7)]
                pygame.draw.polygon(self.screen, WHITE, pts)

            # label below shaft
            self._draw_text(
                self.screen, f"E{i+1}", self.f_tiny, color,
                sx + self.SHAFT_W // 2, TOP_PAD + self.BLDG_H + 6, anchor="center"
            )

    # ── stats panel ───────────────────────────────────────────────────────────
    def _draw_stats(self):
        px = STATS_X

        # title
        self._draw_text(self.screen, f"RL Elevator — {self.agent_name}",
                        self.f_lg, WHITE, px, 10)

        # metric row
        metrics = [
            ("Step",      str(self.step)),
            ("Waiting",   str(self.env.building.total_waiting())),
            ("Delivered", str(self.delivered)),
            ("FPS",       str(int(self.clock.get_fps()))),
        ]
        mx = px
        for label, val in metrics:
            card_w = 110
            self._rounded_rect(self.screen, PANEL_BG, (mx, 36, card_w, 46), radius=6)
            self._draw_text(self.screen, label, self.f_tiny, GRAY, mx + 8, 42)
            self._draw_text(self.screen, val,   self.f_md,  WHITE, mx + 8, 56)
            mx += card_w + 8

        # epsilon bar
        eps = getattr(self.agent, "epsilon", 0)
        bar_y = 96
        self._draw_text(self.screen, f"ε-greedy  {eps:.3f}", self.f_sm, GRAY, px, bar_y)
        bar_bg = (px, bar_y + 16, STATS_W, 6)
        pygame.draw.rect(self.screen, SHAFT_BG, bar_bg, border_radius=3)
        bar_fill = (px, bar_y + 16, int(STATS_W * eps), 6)
        pygame.draw.rect(self.screen, PURPLE, bar_fill, border_radius=3)

        # reward sparkline
        spark_y = 126
        self._draw_text(self.screen, "Reward history", self.f_sm, GRAY, px, spark_y)
        spark_rect = pygame.Rect(px, spark_y + 16, STATS_W, 110)
        self._rounded_rect(self.screen, PANEL_BG, spark_rect, radius=6)

        if len(self.reward_hist) > 2:
            data = self.reward_hist[-STATS_W:]
            mn, mx_ = min(data), max(data)
            rng = mx_ - mn or 1
            pts = []
            for j, v in enumerate(data):
                sx2 = spark_rect.x + j
                sy2 = spark_rect.bottom - int((v - mn) / rng * spark_rect.height) - 2
                pts.append((sx2, sy2))
            if len(pts) > 1:
                pygame.draw.lines(self.screen, GREEN, False, pts, 1)

            # SCAN reference line
            scan_y = spark_rect.bottom - int((self.scan_baseline - mn) / rng * spark_rect.height) - 2
            if spark_rect.top < scan_y < spark_rect.bottom:
                pygame.draw.line(self.screen, SCAN_RED,
                                 (spark_rect.x, scan_y), (spark_rect.right, scan_y), 1)
                self._draw_text(self.screen, "SCAN", self.f_tiny, SCAN_RED,
                                spark_rect.right - 32, scan_y - 10)

        # traffic pattern curve (sinusoidal arrival rate)
        traffic_y = spark_y + 140
        self._draw_text(self.screen, "Traffic pattern (arrival rate)", self.f_sm, GRAY, px, traffic_y)
        t_rect = pygame.Rect(px, traffic_y + 16, STATS_W, 60)
        self._rounded_rect(self.screen, PANEL_BG, t_rect, radius=6)
        t_pts = []
        for tx in range(STATS_W):
            sim_t  = tx * 500 / STATS_W
            rate   = 0.4 + 0.6 * math.sin(math.pi * sim_t / 300) ** 2 + \
                     0.4 * math.sin(math.pi * (sim_t - 250) / 300) ** 2
            ty_val = t_rect.bottom - int((rate / 1.4) * t_rect.height) - 2
            t_pts.append((px + tx, ty_val))
            # mark current position
            cur_x  = int(self.env.building.time / 500 * STATS_W)
        if len(t_pts) > 1:
            pygame.draw.lines(self.screen, AMBER, False, t_pts, 1)
        # vertical cursor
        cur_x = min(int(self.env.building.time / 500 * STATS_W), STATS_W - 1)
        pygame.draw.line(self.screen, WHITE,
                         (px + cur_x, t_rect.top), (px + cur_x, t_rect.bottom), 1)

        # per-floor waiting bars
        bar_sec_y = traffic_y + 90
        self._draw_text(self.screen, "Waiting per floor", self.f_sm, GRAY, px, bar_sec_y)
        bar_sec_y += 16
        row_h   = max(4, (H - bar_sec_y - 60) // config.NUM_FLOORS)

        for f in range(config.NUM_FLOORS - 1, -1, -1):
            count  = len(self.env.building.waiting[f])
            bar_w  = min(count * 14, STATS_W - 30)
            row_y  = bar_sec_y + (config.NUM_FLOORS - 1 - f) * row_h
            # background
            pygame.draw.rect(self.screen, SHAFT_BG,
                             (px, row_y, STATS_W - 30, row_h - 1), border_radius=2)
            # filled bar
            if bar_w > 0:
                col = RED if count > 6 else AMBER if count > 3 else TEAL
                pygame.draw.rect(self.screen, col,
                                 (px, row_y, bar_w, row_h - 1), border_radius=2)

        # priority legend
        leg_y = H - 34
        self._draw_text(self.screen, "Passengers:", self.f_tiny, GRAY, px, leg_y)
        lx = px + 82
        for ptype, col in PRIORITY_COLS.items():
            pygame.draw.circle(self.screen, col, (lx + 4, leg_y + 5), 4)
            self._draw_text(self.screen, ptype.capitalize(), self.f_tiny, GRAY, lx + 12, leg_y)
            lx += 80

        # controls hint bottom right
        hint = "SPACE=speed  R=reset  ESC=quit"
        self._draw_text(self.screen, hint, self.f_tiny, GRAY,
                        W - 10, H - 12, anchor="bottomright")

    # ── main loop ─────────────────────────────────────────────────────────────
    def run(self, max_steps=5000):
        state, _ = self.env.reset()
        running  = True

        while running and self.step < max_steps:
            # ── events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_SPACE:
                        self.fps = 60 if self.fps <= 10 else 10
                    if event.key == pygame.K_r:
                        state, _ = self.env.reset()
                        self.step = 0
                        self.reward_hist.clear()
                        self.delivered = 0

            if not self.paused:
                # ── agent step
                action              = self.agent.get_action(state)
                state, reward, done, _, _ = self.env.step(action)
                self.reward_hist.append(reward)
                self.delivered      = self.env.total_delivered
                self.step          += 1
                if done:
                    state, _ = self.env.reset()

            # ── draw
            self.screen.fill(BG)
            self._draw_building()
            self._draw_elevators()
            self._draw_stats()
            pygame.display.flip()
            self.clock.tick(self.fps)

        pygame.quit()