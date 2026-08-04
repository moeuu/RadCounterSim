import numpy as np

from radcounter.validation.aerial import (
    MultirotorDynamics,
    MultirotorSpec,
    MultirotorState,
    run_rotor_dynamics_gate,
)


def test_quad_hover_command_and_motor_lag_are_physical() -> None:
    spec = MultirotorSpec.quad_x()
    dynamics = MultirotorDynamics(spec)
    commands = dynamics.hover_commands()
    assert np.all((commands > 0.0) & (commands < 1.0))
    state = MultirotorState.at_rest(4)
    stepped = dynamics.step(state, commands, 0.01)
    assert np.all(stepped.rotor_speed_rad_s > 0.0)
    assert np.all(
        stepped.rotor_speed_rad_s
        < np.asarray([rotor.maximum_speed_rad_s for rotor in spec.rotors])
    )


def test_rotor_dynamics_gate_exercises_hover_and_roll() -> None:
    result = run_rotor_dynamics_gate()
    assert result.passed
    assert result.finite_state
    assert result.hover_error_m < 0.2
    assert result.maximum_roll_deg > 4.0
