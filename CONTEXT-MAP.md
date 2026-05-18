# Context Map

## Contexts

- [Lateral Torque](./docs/contexts/lateral-torque/CONTEXT.md) - defines language for versioned lateral torque-control behavior.
- [Lateral Control](./docs/contexts/lateral-control/CONTEXT.md) - defines language for lateral path demand, low-speed lateral behavior, and actuation evidence.

## Relationships

- **Lateral Torque -> Controls**: Lateral torque terms describe how lateral-control behavior is compared, selected, and validated.
- **Lateral Control -> Controls**: Lateral-control terms describe how path demand, low-speed behavior, and actuation evidence are separated before assigning implementation ownership.
- **Lateral Torque -> Lateral Control**: Lateral torque is one actuation boundary that follows processed lateral demand from the broader lateral-control context.
