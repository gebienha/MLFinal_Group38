if stable_char is not None and detected_letter == stable_char:
        #     if stable_start_time and time.time() - stable_start_time >= stability_duration:
        #         # If current character is '-' and nothing has been detected previously
        #         if detected_letter == '-' and not nothing_detected:
        #             nothing_detected = True
        #         # If current character is not '-', or after '-' has been detected, append the detected letter
        #         elif detected_letter != '-' and (nothing_detected or detected_letter != last_char):
        #             sentence.append(detected_letter)
        #             last_char = detected_letter
        #             nothing_detected = False
        #             stable_start_time = None
        # else:
        #     # Update stable_char to the new detected letter and reset timing
        #     stable_char = detected_letter
        #     stable_start_time = time.time()