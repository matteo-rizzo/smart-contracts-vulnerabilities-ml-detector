import json


def extract_and_parse_json(text):
    """ Extracts and parses the first valid JSON object from a string with extra content. """
    try:
        # Find the first balanced JSON object using a brace counter
        start = text.find("{")
        if start == -1:
            raise ValueError("No JSON found in the input text.")

        brace_count = 0
        json_str = ""

        # Iterate over characters to find balanced braces
        for i in range(start, len(text)):
            char = text[i]
            json_str += char

            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1

            # Stop when all braces are balanced
            if brace_count == 0:
                break

        # Parse the extracted JSON string
        parsed_json = json.loads(json_str)
        print(f"Extracted JSON: {json_str}")
        return parsed_json

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        raise
