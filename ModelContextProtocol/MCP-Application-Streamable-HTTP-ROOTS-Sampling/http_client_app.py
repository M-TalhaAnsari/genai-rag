import sys
import json
import gradio as gr
from mcp_http_client_base import MCPHTTPClient


class MCPHTTPClientApp(MCPHTTPClient):
    """GUI client application that extends the base HTTP client with Gradio interface."""

    def __init__(self, server_url: str, roots_dir: str):
        super().__init__(server_url, roots_dir)
        self.tools_cache = []
        self.prompts_cache = []

    async def gui_list_tools(self):
        """List tools for GUI."""
        await self.connect()
        tools = await self.list_tools()
        self.tools_cache = [(t.name, f"{t.name}: {t.description}") for t in tools]
        output = "\n".join([f"• {t.name}: {t.description}" for t in tools])
        choices = [t.name for t in tools]
        return output, gr.update(choices=choices)

    async def gui_call_tool(self, tool_name, arguments_json):
        """Call a tool from GUI."""
        await self.connect()
        if not tool_name:
            return "Error: Please select a tool from the dropdown first"
        try:
            args = json.loads(arguments_json) if arguments_json else {}
            result = await self.call_tool(tool_name, args)
            output = ""
            for content in result.content:
                if hasattr(content, 'text'):
                    output += content.text + "\n"
            return output if output else "No response"
        except json.JSONDecodeError:
            return "Error: Invalid JSON format"
        except Exception as e:
            return f"Error: {e}"

    async def gui_list_resources(self):
        """List resources for GUI."""
        await self.connect()
        resources = await self.list_resources()
        if resources:
            output = []
            for r in resources:
                name = getattr(r, 'name', getattr(r, 'description', 'Unnamed resource'))
                uri_template = getattr(r, 'uriTemplate', getattr(r, 'uri', 'N/A'))
                output.append(f"• {name}\n  URI template: {uri_template}")
            return "\n\n".join(output)
        return "No resources available"

    async def gui_read_resource(self, uri):
        """Read a resource from GUI."""
        await self.connect()
        if not uri:
            return "Error: Please enter a resource URI"
        try:
            result = await self.read_resource(uri)
            output = ""
            for content in result.contents:
                if hasattr(content, 'text'):
                    output += content.text + "\n"
            return output if output else "No content"
        except Exception as e:
            return f"Error: {e}"

    async def gui_list_prompts(self):
        """List prompts for GUI."""
        await self.connect()
        prompts = await self.list_prompts()
        self.prompts_cache = prompts
        output = []
        choices = []
        for p in prompts:
            args_info = ""
            if p.arguments:
                arg_names = [arg.name for arg in p.arguments]
                args_info = f" (args: {', '.join(arg_names)})"
            output.append(f"• {p.name}: {p.description}{args_info}")
            choices.append(p.name)
        return "\n".join(output), gr.update(choices=choices)

    async def gui_get_prompt(self, prompt_name, arguments_json):
        """Get a prompt from GUI."""
        await self.connect()
        if not prompt_name:
            return "Error: Please select a prompt from the dropdown first"
        try:
            args = json.loads(arguments_json) if arguments_json else {}
            result = await self.get_prompt(prompt_name, args)
            output = f"--- Prompt: {result.description} ---\n\n"
            for msg in result.messages:
                content_text = msg.content.text if hasattr(msg.content, 'text') else msg.content.get('text', '')
                output += f"{msg.role}: {content_text}\n"
            return output
        except json.JSONDecodeError:
            return "Error: Invalid JSON format"
        except Exception as e:
            return f"Error: {e}"

    def create_interface(self):
        """Create the Gradio interface."""

        with gr.Blocks(title="MCP HTTP Client") as interface:
            gr.Markdown("# MCP HTTP Client - Remote Server Access")
            gr.Markdown(f"""
            **Server:** {self.server_url}
            **Workspace Roots:** {self.roots_dir}

            This client connects to a remote MCP server via HTTP transport.
            All file operations are restricted to the workspace roots directory.
            """)

            with gr.Tabs():
                with gr.Tab("Tools"):
                    gr.Markdown("### Discover and Execute Server Tools")
                    with gr.Row():
                        with gr.Column():
                            list_tools_btn = gr.Button("List Tools", variant="primary")
                            tools_output = gr.Textbox(label="Available Tools", lines=5)

                        with gr.Column():
                            tool_dropdown = gr.Dropdown(label="Select Tool", choices=[], interactive=True)
                            tool_args = gr.Textbox(
                                label="Arguments (JSON)",
                                placeholder='{"filepath": "test.txt"}',
                                lines=3
                            )
                            call_tool_btn = gr.Button("Call Tool", variant="primary")
                            tool_result = gr.Textbox(label="Tool Result", lines=8)

                    list_tools_btn.click(
                        fn=self.gui_list_tools,
                        outputs=[tools_output, tool_dropdown]
                    )

                    call_tool_btn.click(
                        fn=self.gui_call_tool,
                        inputs=[tool_dropdown, tool_args],
                        outputs=tool_result
                    )

                with gr.Tab("Resources"):
                    gr.Markdown("### Access Server Resources")
                    with gr.Row():
                        with gr.Column():
                            list_resources_btn = gr.Button("List Resource Templates", variant="primary")
                            resources_output = gr.Textbox(label="Available Resources", lines=5)

                        with gr.Column():
                            resource_uri = gr.Textbox(
                                label="Resource URI",
                                placeholder="file://workspace/README.md",
                                lines=1
                            )
                            read_resource_btn = gr.Button("Read Resource", variant="primary")
                            resource_content = gr.Textbox(label="Resource Content", lines=10)

                    list_resources_btn.click(
                        fn=self.gui_list_resources,
                        outputs=resources_output
                    )

                    read_resource_btn.click(
                        fn=self.gui_read_resource,
                        inputs=resource_uri,
                        outputs=resource_content
                    )

                with gr.Tab("Prompts"):
                    gr.Markdown("### List and Get Prompts")
                    with gr.Row():
                        with gr.Column():
                            list_prompts_btn = gr.Button("List Prompts", variant="primary")
                            prompts_output = gr.Textbox(label="Available Prompts", lines=5)

                        with gr.Column():
                            prompt_dropdown = gr.Dropdown(label="Select Prompt", choices=[], interactive=True)
                            prompt_args = gr.Textbox(
                                label="Arguments (JSON)",
                                placeholder='{"filename": "example.py"}',
                                lines=2
                            )
                            get_prompt_btn = gr.Button("Get Prompt", variant="primary")
                            prompt_result = gr.Textbox(label="Prompt Messages", lines=10)

                    list_prompts_btn.click(
                        fn=self.gui_list_prompts,
                        outputs=[prompts_output, prompt_dropdown]
                    )

                    get_prompt_btn.click(
                        fn=self.gui_get_prompt,
                        inputs=[prompt_dropdown, prompt_args],
                        outputs=prompt_result
                    )

        return interface



def main():
    if len(sys.argv) < 3:
        print("Usage: python mcp_http_client_app.py <server_url> <roots_dir>")
        print("Example: python mcp_http_client_app.py http://127.0.0.1:8000 /path/to/workspace")
        sys.exit(1)

    server_url = sys.argv[1]
    roots_dir = sys.argv[2]

    client = MCPHTTPClientApp(server_url, roots_dir)
    interface = client.create_interface()
    interface.queue().launch(server_name="127.0.0.1", server_port=7861)


if __name__ == "__main__":
    main()
