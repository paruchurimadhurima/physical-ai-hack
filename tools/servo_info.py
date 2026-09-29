"""Read-only diagnostics for Feetech/Waveshare STS/ST bus servos.

Scans the bus for servos and prints model number, firmware, voltage limits,
the voltage the servo currently measures, and any error flags. Never writes
anything to the servo.

Usage:
    python tools/servo_info.py /dev/tty.usbmodem5B8E1150551
"""

import sys

import scservo_sdk as scs

BAUDRATES = [1_000_000, 500_000, 250_000, 115_200]
SCAN_IDS = range(0, 21)

ADDR_FIRMWARE_MAJOR = 0
ADDR_FIRMWARE_MINOR = 1
ADDR_MAX_VOLTAGE = 14
ADDR_MIN_VOLTAGE = 15
ADDR_PRESENT_VOLTAGE = 62
ADDR_PRESENT_TEMPERATURE = 63

ERROR_BITS = {
    scs.ERRBIT_VOLTAGE: "VOLTAGE (Spannung ausserhalb der Grenzen!)",
    scs.ERRBIT_ANGLE: "ANGLE",
    scs.ERRBIT_OVERHEAT: "OVERHEAT",
    scs.ERRBIT_OVERELE: "OVERCURRENT",
    scs.ERRBIT_OVERLOAD: "OVERLOAD",
}


def read1(ph, port, servo_id, addr):
    value, result, _ = ph.read1ByteTxRx(port, servo_id, addr)
    return value if result == scs.COMM_SUCCESS else None


def volts(raw):
    return "?" if raw is None else f"{raw / 10:.1f} V"


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)

    port = scs.PortHandler(sys.argv[1])
    ph = scs.PacketHandler(0)
    if not port.openPort():
        sys.exit(f"Port {sys.argv[1]} konnte nicht geoeffnet werden (belegt oder nicht angesteckt?).")

    found = 0
    try:
        for baudrate in BAUDRATES:
            port.setBaudRate(baudrate)
            for servo_id in SCAN_IDS:
                model, result, error = ph.ping(port, servo_id)
                if result != scs.COMM_SUCCESS:
                    continue
                found += 1
                errors = [name for bit, name in ERROR_BITS.items() if error & bit]
                print(f"Servo gefunden: ID={servo_id} @ {baudrate} baud")
                print(f"  Modellnummer   : {model}  (lerobot erwartet fuer sts3215: 777)")
                print(
                    f"  Firmware       : {read1(ph, port, servo_id, ADDR_FIRMWARE_MAJOR)}"
                    f".{read1(ph, port, servo_id, ADDR_FIRMWARE_MINOR)}"
                )
                print(f"  Spannung jetzt : {volts(read1(ph, port, servo_id, ADDR_PRESENT_VOLTAGE))}")
                print(
                    f"  Erlaubt        : {volts(read1(ph, port, servo_id, ADDR_MIN_VOLTAGE))}"
                    f" .. {volts(read1(ph, port, servo_id, ADDR_MAX_VOLTAGE))}"
                )
                print(f"  Temperatur     : {read1(ph, port, servo_id, ADDR_PRESENT_TEMPERATURE)} C")
                print(f"  Fehler-Flags   : {', '.join(errors) if errors else 'keine'}")
        if not found:
            print("Kein Servo gefunden. Netzteil an? Kabel richtig eingerastet?")
    finally:
        port.closePort()


if __name__ == "__main__":
    main()
