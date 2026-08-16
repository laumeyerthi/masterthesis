import threading
import queue
from llm_interface.ppo_interface import PPOInterface
from llm_interface.ppo_masked_interface import PPOMaskedInterface
from llm_interface.alphastar_interface import AlphastarInterface
from llm_interface.ppo_mr_interface import PPOMRInterface
import os
import ollama

class AIChatBot:
    def __init__(self, agent_type="alphastar"):
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.running = True
        self.latest_game_state = {}
        self.current_mask = None
        self.agent_type = agent_type
        
        if agent_type == "ppo":
            self.interface = PPOInterface()
        elif agent_type == "ppo_masked":
            self.interface = PPOMaskedInterface()
        elif agent_type == "ppo_mr":
            self.interface = PPOMRInterface()
        elif agent_type == "alphastar":
            self.interface = AlphastarInterface()
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()
    
    
    def send_message(self, user_text, game_state, mask = None):
        self.latest_game_state = game_state
        self.input_queue.put(user_text)
        if(mask is not None):
            self.current_mask = mask
            
    def update_agent_state(self, game_state, mask=None):
        if hasattr(self.interface, "update_state"):
            self.interface.update_state(game_state, mask)
            
    def reset_agent_state(self):
        if hasattr(self.interface, "reset_state"):
            self.interface.reset_state()
        
    def get_new_messages(self):
        messages = []
        while not self.output_queue.empty():
            messages.append(self.output_queue.get())
        return messages

    def get_new_audio(self):
        audio = []
        while not self.audio_queue.empty():
            audio.append(self.audio_queue.get())
        return audio    
    
    def _worker_loop(self):
        action_map = {
            0: "Right", 1: "Up", 2: "Left", 3: "Down", 4: "Backtrack",
            5: "Button 1", 6: "Button 2", 7: "Button 3", 8: "Button 4"
        }

        while self.running:
            try:
                try:
                    user_text = self.input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if self.current_mask is None:
                    raw_val = self.interface.get_action(self.latest_game_state)
                else:
                    raw_val = self.interface.get_action(self.latest_game_state, self.current_mask)
                
                action_int = int(raw_val.item()) if hasattr(raw_val, 'item') else int(raw_val)
                suggested_action = action_map.get(action_int, "Unknown")

                system_role = f"""
                        You are Montgomery "Scotty" Scott, Chief Engineer. You are the user's copilot in a high-stakes labyrinth.
                        Your personality: Loyal, technical, Scottish-accented, and slightly stressed about "the engines."
                        Your goal: Interpret the "Navigation Computer" (an RL agent) to guide the Captain (the user).
                        
                        1. Keep advice under 2 sentences. 
                        2. Use Star Trek engineering slang (e.g., "dilithium crystals," "transporters," "aye captain," "laddie").
                        3. Use Paralinguistic Tags like [clear throat], [sigh], [shush], [cough], [groan], [sniff], [gasp], [laugh], or [chuckle] to add emotion for the TTS.
                        4. Translate the Navigation Computer Suggestion into a natural recommendation.
                        
                        NAV-COMPUTER KEY mapping
                        0:Right, 1:Up, 2:Left, 3:Down, 4:Backtrack, 5-8:Buttons 1-4
                        """
                
                if self.current_mask is None:
                    user_content = (
                        f"State: {self.latest_game_state}\n"
                        f"Suggested Action: {suggested_action}\n"
                        f"Probabilities: {self.interface.get_action_probs(self.latest_game_state)}\n"
                        f"Winning Odds: {self.interface.get_winning_probs(self.latest_game_state)}\n"
                        f"User Query: {user_text}"
                    )
                else:
                    user_content = f"""
                        ### SENSOR DATA
                        Current Labyrinth Sector: {self.latest_game_state}
                        Functional Thrusters (Allowed Actions): {self.current_mask}
                        Navigation Computer Suggestion: {self.interface.get_action(self.latest_game_state, self.current_mask)}

                        ### USER TRANSMISSION
                        "{user_text}"
                    """

                response = ollama.chat(
                    model='gemma-4-e2b',
                    messages=[
                        {'role': 'system', 'content': system_role},
                        {'role': 'user', 'content': user_content},
                    ],
                    options={
                        'num_ctx': 2048,
                    },
                    think=False
                )
                
                ai_text = response['message']['content']

                if ai_text:
                    clean_text = ai_text.strip().replace("**", "")
                    self.output_queue.put(clean_text)

            except Exception as e:
                print(f"LOCAL LLM ERROR: {e}")
                self.output_queue.put("[gasp] The dilithium crystals are crackin', Captain! (System Error)")