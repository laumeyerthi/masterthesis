import threading
import queue
from google import genai
from google.genai import types
from llm_interface.ppo_interface import PPOInterface
from llm_interface.ppo_masked_interface import PPOMaskedInterface
from llm_interface.alphastar_interface import AlphastarInterface
from llm_interface.ppo_mr_interface import PPOMRInterface
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY=os.getenv("GEMINI_API_KEY")

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
        if not API_KEY:
            raise ValueError("GEMINI_API_KEY not found! Check your .env file.")
        self.client = genai.Client(api_key=API_KEY)
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
        try:
            chat_config = types.GenerateContentConfig(
                response_modalities=["TEXT", "AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name="Fenrir" # Puck, Charon, Kore, Fenrir, Aoede, Leda, Orus, and Zephyr
                        )
                    )
                )
            )
            chat = self.client.chats.create(
                #model="gemini-2.5-pro",
                model = "gemini-3.1-flash-lite"
                #config=chat_config
            )  
        except Exception as e:
            print(f"Failed to create chat: {e}")
            return
        while self.running:
            try:
                try:
                    user_text = self.input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                context_str = ""
                if self.current_mask is None:
                    context_str = f"""
                        [SYSTEM CONTEXT]
                        Current Game State: {self.latest_game_state}
                        Current Action Prediction : {self.interface.get_action(self.latest_game_state)}
                        Current Action Propabilities: {self.interface.get_action_probs(self.latest_game_state)}
                        Current Advantages : {self.interface.get_winning_probs(self.latest_game_state)}
                        User Input: {user_text}
                        
                        Instruction: You are a helpful game copilot to a labirynth game. You have the results from an ppo rl agent trained on the game. Keep advice short (under 2 sentences). The actions are in order right, up, left, down, backtrack, button1, button2, button3 and button4.
                        """
                else:
                    # context_str = f"""
                    #     [SYSTEM CONTEXT]
                    #     Current Game State: {self.latest_game_state}
                    #     Current Allowed Actions: {self.current_mask}
                    #     Current Action Prediction : {self.interface.get_action(self.latest_game_state, self.current_mask)}
                    #     User Input: {user_text}
                        
                    #     Instruction: You are a helpful game copilot to a labirynth game. You have the results from an ppo rl agent trained on the game. Keep advice short (under 2 sentences). The actions are in order right, up, left, down, backtrack, button1, button2, button3 and button4.
                    #     """
                    # Current Action Prediction : {self.interface.get_action(self.latest_game_state, self.current_mask)}
                    # Current Action Propabilities: {self.interface.get_action_probs(self.latest_game_state, self.current_mask)} 
                    # Current Advantages : {self.interface.get_winning_probs(self.latest_game_state)}
                    context_str = f"""
                        ### SYSTEM ROLE
                        You are Montgomery "Scotty" Scott, Chief Engineer. You are the user's copilot in a high-stakes labyrinth.
                        Your personality: Loyal, technical, Scottish-accented, and slightly stressed about "the engines."
                        Your goal: Interpret the "Navigation Computer" (an RL agent) to guide the Captain (the user).

                        ### SENSOR DATA
                        Current Labyrinth Sector: {self.latest_game_state}
                        Functional Thrusters (Allowed Actions): {self.current_mask}
                        Navigation Computer Suggestion: {self.interface.get_action(self.latest_game_state, self.current_mask)}

                        ### NAV-COMPUTER KEY
                        0:Right, 1:Up, 2:Left, 3:Down, 4:Backtrack, 5-8:Buttons 1-4

                        ### USER TRANSMISSION
                        "{user_text}"

                        ### INSTRUCTIONS
                        1. Keep advice under 2 sentences. 
                        2. Use Star Trek engineering slang (e.g., "dilithium crystals," "transporters," "aye captain," "laddie").
                        3. Use Paralinguistic Tags like [clear throat], [sigh], [shush], [cough], [groan], [sniff], [gasp], [laugh], or [chuckle] to add emotion for the TTS.
                        4. Translate the Navigation Computer Suggestion into a natural recommendation.
                        """
                response = chat.send_message(context_str)
                   
                ai_text = response.text
                if ai_text:
                    clean_text = ai_text.strip().replace("**", "")
                    self.output_queue.put(f"{clean_text}")
                    
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if part.inline_data and part.inline_data.mime_type.startswith("audio/"):
                            audio_bytes = part.inline_data.data
                            self.audio_queue.put(audio_bytes)
            except Exception as e:
                print(f"API ERROR: {e}")
                self.output_queue.put("System: AI Error (See Console)")