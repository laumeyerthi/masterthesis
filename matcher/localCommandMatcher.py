import difflib

class LocalCommandMatcher:
    def __init__(self):
        self.COMMAND_MAP = {
            "stop": "STOP", "halt": "STOP", "stoppen": "STOP", "anhalten": "STOP",
            "go": "MOVE", "geh": "MOVE", "laufen": "MOVE", "move": "MOVE"
        }
        self.valid_commands = list(self.COMMAND_MAP.keys())
        
    def process_input(self, user_text, cutoff=0.75):
        words = user_text.lower().split()
        
        for word in words:
            matches = difflib.get_close_matches(word, self.valid_commands, n=1, cutoff=cutoff)
            
            if matches:
                best_match = matches[0]
                return self.COMMAND_MAP[best_match]
                
        return "NO_MATCH"