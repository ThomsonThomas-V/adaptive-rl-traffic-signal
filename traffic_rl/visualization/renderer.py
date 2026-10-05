"""Simple top-down Pygame renderer for the intersection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pygame

    from traffic_rl.envs.traffic_env import TrafficLightEnv


class TrafficRenderer:
    """Render roads, signal state, queues, and live metrics."""

    width = 720
    height = 720

    def __init__(self, mode: str = "human") -> None:
        import pygame

        self.mode = mode
        pygame.init()
        pygame.font.init()
        self.surface: pygame.Surface
        if mode == "human":
            self.surface = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Adaptive Traffic Signal Control")
        else:
            self.surface = pygame.Surface((self.width, self.height))
        self.font = pygame.font.SysFont("consolas", 20)
        self.clock = pygame.time.Clock()

    def render(self, env: TrafficLightEnv) -> NDArray[np.uint8] | None:
        import pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
        self.surface.fill((43, 120, 70))
        road = (65, 67, 70)
        pygame.draw.rect(self.surface, road, (260, 0, 200, self.height))
        pygame.draw.rect(self.surface, road, (0, 260, self.width, 200))
        pygame.draw.rect(self.surface, (95, 95, 95), (260, 260, 200, 200))
        for x1, y1, x2, y2 in ((360, 0, 360, 250), (360, 470, 360, 720), (0, 360, 250, 360), (470, 360, 720, 360)):
            pygame.draw.line(self.surface, (225, 210, 80), (x1, y1), (x2, y2), 3)

        yellow = env.in_yellow
        ns_green = not yellow and int(env.current_phase) == 0
        ew_green = not yellow and int(env.current_phase) == 1
        ns_color = (245, 190, 35) if yellow else ((30, 220, 80) if ns_green else (220, 45, 45))
        ew_color = (245, 190, 35) if yellow else ((30, 220, 80) if ew_green else (220, 45, 45))
        for center, color in (((235, 235), ns_color), ((485, 485), ns_color), ((485, 235), ew_color), ((235, 485), ew_color)):
            pygame.draw.circle(self.surface, color, center, 15)

        # Cars are compact queue markers, capped visually while numeric labels stay exact.
        origins = ((330, 235, 0, -1), (390, 485, 0, 1), (485, 330, 1, 0), (235, 390, -1, 0))
        for queue, (x, y, dx, dy) in zip(env.queues, origins):
            for index in range(min(int(queue), 12)):
                rect = pygame.Rect(x + dx * index * 18 - 7, y + dy * index * 18 - 11, 14, 22)
                if dx:
                    rect = pygame.Rect(x + dx * index * 18 - 11, y - 7, 22, 14)
                pygame.draw.rect(self.surface, (65, 155, 235), rect, border_radius=2)

        phase_text = "YELLOW" if yellow else env.current_phase.name
        lines = (
            f"t={env.timestep:4d}  phase={phase_text}",
            f"queues N/S/E/W: {env.queues.tolist()}",
            f"processed: {env.total_processed}  switches: {env.signal_switches}",
            f"reward: {env.last_reward:8.2f}",
        )
        for index, line in enumerate(lines):
            self.surface.blit(self.font.render(line, True, (245, 245, 245)), (15, 15 + 26 * index))

        if self.mode == "human":
            pygame.display.flip()
            self.clock.tick(10)
            return None
        pixels = pygame.surfarray.array3d(self.surface)
        return np.transpose(pixels, (1, 0, 2)).copy()

    def close(self) -> None:
        import pygame

        pygame.quit()
