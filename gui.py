import tkinter as tk
import customtkinter as ctk
import time
import threading
from main_app import MultiAgentSystem

# Set appearance mode and default color theme for customtkinter
ctk.set_appearance_mode("Dark")  # Setting default to Dark mode
ctk.set_default_color_theme("blue")

class ChatMessage(ctk.CTkFrame):
    """A custom widget for chat messages with rounded corners and proper styling"""
    def __init__(self, master, message, is_user=False, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        
        # Configure message appearance based on sender
        if is_user:
            msg_color = "#2B5278"  # Darker blue for user messages
            text_color = "#FFFFFF"  # White text for user
            anchor_position = "e"  # East (right) alignment for user messages
        else:
            msg_color = "#383838"  # Dark gray for bot messages
            text_color = "#FFFFFF"  # White text for bot
            anchor_position = "w"  # West (left) alignment for bot messages
        
        # Create a container for the message bubble
        self.bubble_frame = ctk.CTkFrame(
            self, 
            fg_color=msg_color,
            corner_radius=12
        )
        
        # Create message label with proper wrapping
        self.message_label = ctk.CTkLabel(
            self.bubble_frame,
            text=message,
            text_color=text_color,
            wraplength=400,  # Maximum width of the message
            justify=tk.LEFT,
            anchor="w",
            padx=12,
            pady=8,
            font=("Arial", 13)
        )
        self.message_label.pack(padx=0, pady=0, fill="both", expand=True)
        
        # Position the bubble according to sender
        self.bubble_frame.pack(padx=20, pady=5, anchor=anchor_position, fill="x", expand=False)

class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Learning Assistant Chatbot")
        self.root.geometry("800x600")
        self.root.minsize(600, 400)
        
        # Set the dark theme colors
        self.bg_color = "#1E1E1E"  # Dark background
        self.header_color = "#252526"  # Slightly lighter for header
        self.input_area_color = "#252526"  # Same as header
        self.text_color = "#FFFFFF"  # White text
        
        # Create custom fonts
        self.header_font = ctk.CTkFont(family="Arial", size=16, weight="bold")
        self.normal_font = ctk.CTkFont(family="Arial", size=13)
        self.input_font = ctk.CTkFont(family="Arial", size=14)
        
        # Initialize the chatbot system
        self.system = MultiAgentSystem()
        
        # Create a main frame
        self.main_frame = ctk.CTkFrame(root, fg_color=self.bg_color, corner_radius=0)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create header
        self.header = ctk.CTkFrame(self.main_frame, fg_color=self.header_color, height=60, corner_radius=0)
        self.header.pack(fill=tk.X, side=tk.TOP)
        
        self.header_label = ctk.CTkLabel(
            self.header, 
            text="Learning Assistant Chatbot",
            font=self.header_font,
            text_color=self.text_color
        )
        self.header_label.pack(pady=12)
        
        # Create the main content area with chat messages
        self.chat_frame = ctk.CTkScrollableFrame(
            self.main_frame, 
            fg_color=self.bg_color,
            scrollbar_fg_color=self.header_color
        )
        self.chat_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # Typing indicator (hidden initially)
        self.typing_frame = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        self.typing_label = ctk.CTkLabel(
            self.typing_frame,
            text="Bot is typing...",
            font=("Arial", 12, "italic"),
            text_color="#AAAAAA"  # Light gray text
        )
        
        # Input area at the bottom
        self.input_frame = ctk.CTkFrame(self.main_frame, fg_color=self.input_area_color, height=80)
        self.input_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Create a container for input elements
        self.input_container = ctk.CTkFrame(self.input_frame, fg_color="#333333", corner_radius=12)
        self.input_container.pack(padx=20, pady=15, fill=tk.X)
        
        # User input field
        self.user_input = ctk.CTkEntry(
            self.input_container,
            placeholder_text="Send a message...",
            font=self.input_font,
            fg_color="#333333",
            text_color=self.text_color,
            placeholder_text_color="#AAAAAA",
            border_width=0,
            height=40
        )
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(15, 10), pady=5)
        self.user_input.bind("<Return>", self.send_message)
        
        # Send button
        self.send_button = ctk.CTkButton(
            self.input_container,
            text="Send",
            font=self.normal_font,
            width=70,
            height=35,
            corner_radius=8,
            fg_color="#4A6EA9",  # Blue button
            hover_color="#5B80B2",  # Lighter blue on hover
            command=self.send_message
        )
        self.send_button.pack(side=tk.RIGHT, padx=(5, 10), pady=5)

        # Focus on input
        self.user_input.focus_set()
    
    def send_message(self, event=None):
        """Handle sending user message and getting response"""
        prompt = self.user_input.get().strip()
        if not prompt:
            return
        
        # Display user message
        self._add_message(prompt, is_user=True)
        
        # Clear input
        self.user_input.delete(0, tk.END)
        
        # Show typing indicator
        self.typing_frame.pack(padx=20, pady=5, anchor="w", fill="x")
        self.typing_label.pack(padx=0, pady=0, anchor="w")
        
        # Get response in a separate thread to avoid UI freezing
        threading.Thread(target=self._get_response, args=(prompt,), daemon=True).start()
    
    def _get_response(self, prompt):
        """Get bot response in a separate thread"""
        # Simulate typing delay (you can remove this if you want immediate responses)
        time.sleep(1)
        
        try:
            # Get agent's response
            response = self.system.handle_user_prompt(prompt)
            
            # Update UI in the main thread
            self.root.after(0, lambda: self._show_response(response))
        except Exception as e:
            # If there's an error, show it as the response
            self.root.after(0, lambda: self._show_response(f"Error: {str(e)}"))
    
    def _show_response(self, response):
        self.typing_frame.pack_forget()

    # Handle multiple messages
        if isinstance(response, list):
            for i, message in enumerate(response):
                self.root.after(i * 100, lambda m=message: self._add_message(m, is_user=False))
        else:
            self._add_message(response, is_user=False)
    
    def _add_message(self, message, is_user=False):
        msg = ChatMessage(self.chat_frame, message, is_user=is_user)
        msg.pack(fill="x", padx=10, pady=5, anchor="w" if not is_user else "e")
        self.chat_frame._parent_canvas.yview_moveto(1.0)



# Run the GUI
if __name__ == "__main__":
    root = ctk.CTk()
    app = ChatbotGUI(root)
    root.mainloop()