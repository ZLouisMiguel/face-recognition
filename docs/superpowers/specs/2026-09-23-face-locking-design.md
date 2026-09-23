# Face Tracking with Identity Lock

## Goal

Deliver the assignment's vision-only face-tracking system: acquire a known face, keep the overlay associated with that identity across normal frame-to-frame changes, survive short detection gaps, and safely reacquire or unlock when the target is gone.

The Arduino sketch remains separate because the announcement explicitly defers motor-based correction.

## Current root causes

- The recognition loop stores boxes, embeddings, and names in separate lists. If alignment or embedding fails for one detection, `zip(names, embeddings, boxes)` associates later embeddings with the wrong boxes.
- The tracker has no acquisition debounce, loss grace period, explicit state, or reset/reacquisition path.
- The tracker matches only appearance similarity and can accept a different but similar-looking face.
- MediaPipe is configured for one face and returns its first face for every YuNet detection.
- The overlay decides lock color using substring matching, so `UNLOCKED` is incorrectly considered locked.

## Design

YuNet detections become records containing a bounding box and the detector's own five landmarks. Each record is processed independently into one `TrackCandidate` containing the box, embedding, recognized identity, and confidence. The recognition loop passes those records to `TargetTracker` rather than maintaining parallel arrays.

`TargetTracker` exposes `SEARCHING`, `LOCKED`, and `LOST` states. A known identity must be observed consistently for a small number of frames before acquisition. Once locked, candidates with a conflicting known identity are rejected; matching candidates reset the missing counter. Short gaps enter `LOST` without immediately destroying the lock, and prolonged gaps clear the lock so a later face must be reacquired.

The landmark fallback selects the MediaPipe face closest to the requested bounding box and supports multiple faces. If YuNet provides valid landmarks, those are preferred because they are already associated with the detection.

## Error handling and compatibility

- Empty or malformed database identities are ignored rather than crashing startup.
- Database saves work for both `data/database.json` and a filename without a directory.
- Existing `detect_faces`, `lock_target`, and `update_lock` compatibility behavior is retained where practical for the manual scripts.
- No serial or motor behavior is added to the live vision loop.

## Verification

Automated `unittest` coverage will exercise lock acquisition, skipped/invalid candidates, identity rejection, temporary loss, reset after prolonged loss, overlay state colors, and database edge cases. Existing interactive scripts remain manual hardware checks. The final verification will include compilation, the automated suite, and a camera walkthrough documented in the README.
