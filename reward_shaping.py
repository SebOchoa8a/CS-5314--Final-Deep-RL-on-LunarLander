"""
Potential-Based Reward Shaping for LunarLander-v3.

Ng, Harada & Russell (1999) — "Policy Invariance Under Reward Transformations:
Theory and Application to Reward Shaping"
Proceedings of the 16th ICML, pp. 278-287.

Core theorem (Theorem 1):
  Any reward shaping function of the form
      F(s, a, s') = gamma * Phi(s') - Phi(s)
  is policy-invariant: the set of optimal policies in the shaped MDP is
  identical to the set of optimal policies in the original MDP.

  Shaped reward:
      r'(s, a, s') = r(s, a, s') + F(s, a, s')
                   = r(s, a, s') + gamma * Phi(s') - Phi(s)

Potential function for LunarLander-v3:
  LunarLander state: [x, y, vx, vy, angle, angular_vel, leg_l, leg_r]
    x, y          — position (landing pad is at origin)
    vx, vy        — linear velocity
    angle         — tilt from upright (0 = vertical)
    angular_vel   — rotational speed
    leg_l, leg_r  — contact booleans (1.0 if touching ground)

  Phi(s) is a weighted sum of four sub-potentials, each encoding a
  property that good landing states satisfy:

    Phi_proximity(s) = -w_prox  * sqrt(x^2 + y^2)
        Reward being close to the landing pad (origin).

    Phi_speed(s)     = -w_speed * sqrt(vx^2 + vy^2)
        Reward low total speed — encourages a gentle approach/landing.

    Phi_angle(s)     = -w_angle * |angle|
        Reward upright orientation — penalises excessive tilt.

    Phi_legs(s)      =  w_legs  * (leg_l + leg_r)
        Reward leg contact — both legs touching = stable on ground.

  Total: Phi(s) = Phi_proximity + Phi_speed + Phi_angle + Phi_legs

  All weights are non-negative and tunable; defaults are chosen so that
  the shaping signal is small relative to typical extrinsic rewards
  (~100-300 for a successful LunarLander episode), preventing the agent
  from exploiting the shaping bonus instead of pursuing the real goal.

Usage (standalone — no changes to existing files required):

    from reward_shaping import PotentialShaping

    shaper = PotentialShaping(gamma=0.99)

    # Inside the step loop, after env.step():
    F = shaper.shaping_bonus(state, next_state)
    r_shaped = r_extrinsic + F

    # Or compute Phi directly for logging / debugging:
    phi = shaper.potential(state)
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# State indices — named constants so the code is self-documenting
# ---------------------------------------------------------------------------

IDX_X           = 0   # horizontal position  (0 = pad centre)
IDX_Y           = 1   # vertical position    (0 = ground level)
IDX_VX          = 2   # horizontal velocity
IDX_VY          = 3   # vertical velocity
IDX_ANGLE       = 4   # tilt angle           (0 = upright)
IDX_ANG_VEL     = 5   # angular velocity
IDX_LEG_LEFT    = 6   # left  leg contact    (1.0 = touching)
IDX_LEG_RIGHT   = 7   # right leg contact    (1.0 = touching)


# ---------------------------------------------------------------------------
# Individual sub-potential functions
# ---------------------------------------------------------------------------

def phi_proximity(state: np.ndarray, w: float = 1.0) -> float:
    """
    Phi_proximity = -w * sqrt(x^2 + y^2)

    Encodes distance to the landing pad. Higher (less negative) when
    the lander is close to the origin.
    """
    x = float(state[IDX_X])
    y = float(state[IDX_Y])
    return -w * math.sqrt(x * x + y * y)


def phi_speed(state: np.ndarray, w: float = 0.5) -> float:
    """
    Phi_speed = -w * sqrt(vx^2 + vy^2)

    Encodes total speed. Higher (less negative) when the lander moves
    slowly — encourages a controlled, gentle descent.
    """
    vx = float(state[IDX_VX])
    vy = float(state[IDX_VY])
    return -w * math.sqrt(vx * vx + vy * vy)


def phi_angle(state: np.ndarray, w: float = 0.5) -> float:
    """
    Phi_angle = -w * |angle|

    Encodes tilt from vertical. Higher (less negative) when the lander
    is upright. Prevents the agent from drifting into extreme tilts.
    """
    return -w * abs(float(state[IDX_ANGLE]))


def phi_legs(state: np.ndarray, w: float = 0.3) -> float:
    """
    Phi_legs = w * (leg_left + leg_right)

    Encodes leg-ground contact. Ranges from 0 (airborne) to 2w (both
    legs down). Provides a small nudge toward stable ground contact.
    """
    return w * (float(state[IDX_LEG_LEFT]) + float(state[IDX_LEG_RIGHT]))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PotentialShaping:
    """
    Wraps all sub-potentials into a single shaping object.

    Parameters
    ----------
    gamma       : discount factor — must match the one used by the agent
                  (Ng et al. 1999 require the same gamma for policy invariance)
    w_proximity : weight for distance-to-pad potential
    w_speed     : weight for speed penalty potential
    w_angle     : weight for tilt penalty potential
    w_legs      : weight for leg-contact bonus potential
    """

    def __init__(
        self,
        gamma: float = 0.99,
        w_proximity: float = 1.0,
        w_speed: float = 0.5,
        w_angle: float = 0.5,
        w_legs: float = 0.3,
    ):
        self.gamma = gamma
        self.w_proximity = w_proximity
        self.w_speed = w_speed
        self.w_angle = w_angle
        self.w_legs = w_legs

    def potential(self, state: np.ndarray) -> float:
        """
        Compute Phi(s) — the total potential of a single state.

        Phi(s) = Phi_proximity(s) + Phi_speed(s) + Phi_angle(s) + Phi_legs(s)
        """
        return (
            phi_proximity(state, self.w_proximity)
            + phi_speed(state, self.w_speed)
            + phi_angle(state, self.w_angle)
            + phi_legs(state, self.w_legs)
        )

    def shaping_bonus(self, state: np.ndarray, next_state: np.ndarray) -> float:
        """
        Compute the potential-based shaping bonus for one transition.

        F(s, s') = gamma * Phi(s') - Phi(s)     (Ng et al. 1999, eq. 1)

        Positive F means the agent moved to a "better" state.
        Negative F means the agent moved to a "worse" state.

        Returns a plain Python float — add directly to extrinsic reward:
            r_shaped = r_extrinsic + shaper.shaping_bonus(s, s_next)
        """
        return self.gamma * self.potential(next_state) - self.potential(state)

    def components(self, state: np.ndarray) -> dict:
        """
        Return each sub-potential individually (useful for logging/ablations).

        Example:
            info = shaper.components(state)
            # {'proximity': -0.42, 'speed': -0.18, 'angle': -0.07, 'legs': 0.0}
        """
        return {
            "proximity": phi_proximity(state, self.w_proximity),
            "speed":     phi_speed(state, self.w_speed),
            "angle":     phi_angle(state, self.w_angle),
            "legs":      phi_legs(state, self.w_legs),
        }
