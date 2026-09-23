import platform
import cv2


class Camera:
    def __init__(self, source=0):
        # cv2.CAP_DSHOW only helps on Windows; it breaks capture on macOS/Linux.
        if platform.system() == "Windows":
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera source {source}. "
                f"Try a different index (0, 1, 2...) or check permissions."
            )

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        return frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
