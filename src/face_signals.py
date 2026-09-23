import cv2


class OutputSignalManager:
    def __init__(self):
        self.status = "UNLOCKED"

    def update_status(self, is_locked, identity_name=None):
        if is_locked:
            self.status = f"LOCKED: {identity_name}"
        else:
            self.status = "SEARCHING / UNLOCKED"

    def draw_overlay(self, frame, box=None):
        color = (0, 255, 0) if "LOCKED" in self.status else (0, 0, 255)

        if box is not None:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame, self.status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        cv2.putText(
            frame,
            f"System Status: {self.status}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
        return frame
