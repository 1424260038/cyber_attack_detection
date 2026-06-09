"""Windowed launcher for the CyberDD demo system."""

from __future__ import annotations

import os
import subprocess
import sys
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import messagebox


PROJECT_DIR = Path(__file__).resolve().parent
WEB_DIR = PROJECT_DIR / "web"
CREATE_NEW_CONSOLE = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)


def python_command() -> str:
    """Return a console Python executable for child commands."""
    if getattr(sys, "frozen", False):
        return "python"

    executable = Path(sys.executable)
    if executable.name.lower() == "pythonw.exe":
        python_exe = executable.with_name("python.exe")
        if python_exe.exists():
            return str(python_exe)
    return str(executable)


PYTHON = python_command()


def command_line(parts: list[str | Path]) -> str:
    return subprocess.list2cmdline([str(part) for part in parts])


def start_shell(title: str, command: str, cwd: Path = PROJECT_DIR) -> None:
    if not cwd.exists():
        messagebox.showerror("路径不存在", f"目录不存在：\n{cwd}")
        return

    if os.name == "nt":
        shell_command = f'title {title} && cd /d "{cwd}" && {command}'
        subprocess.Popen(
            ["cmd.exe", "/k", shell_command],
            cwd=str(cwd),
            creationflags=CREATE_NEW_CONSOLE,
        )
    else:
        subprocess.Popen(command, cwd=str(cwd), shell=True)


class CyberDDLauncher(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("CyberDD 网络攻击行为识别系统启动器")
        self.geometry("760x560")
        self.minsize(680, 500)
        self.configure(bg="#0f172a")

        self.status_var = tk.StringVar(value="就绪：建议先启动后端，再启动前端。")
        self._build_ui()

    def _build_ui(self) -> None:
        header = tk.Frame(self, bg="#0f172a")
        header.pack(fill="x", padx=28, pady=(26, 16))

        tk.Label(
            header,
            text="CyberDD 大创项目启动器",
            font=("Microsoft YaHei UI", 22, "bold"),
            fg="#f8fafc",
            bg="#0f172a",
        ).pack(anchor="w")
        tk.Label(
            header,
            text="多模态深度学习驱动的网络攻击行为识别研究演示入口",
            font=("Microsoft YaHei UI", 11),
            fg="#94a3b8",
            bg="#0f172a",
        ).pack(anchor="w", pady=(8, 0))

        body = tk.Frame(self, bg="#0f172a")
        body.pack(fill="both", expand=True, padx=28)

        left = tk.Frame(body, bg="#111827", highlightthickness=1, highlightbackground="#1f2937")
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        right = tk.Frame(body, bg="#111827", highlightthickness=1, highlightbackground="#1f2937")
        right.pack(side="right", fill="both", expand=True, padx=(10, 0))

        self._section_title(left, "答辩演示")
        self._button(left, "一键启动演示（后端 + 前端）", self.start_demo, primary=True)
        self._button(left, "启动 API 后端", self.start_api)
        self._button(left, "启动 Web 前端", self.start_web)
        self._button(left, "打开 WebUI 页面", self.open_web)
        self._button(left, "打开 API 文档", self.open_api_docs)

        self._section_title(right, "项目维护")
        self._button(right, "运行完整质量检查", self.run_checks)
        self._button(right, "刷新演示系统", self.build_demo_system)
        self._button(right, "生成项目报告", self.generate_report)
        self._button(right, "生成发布包", self.package_release)
        self._button(right, "打开项目目录", self.open_project_dir)
        self._button(right, "打开命令行菜单 run.cmd", self.open_cmd_menu)

        footer = tk.Frame(self, bg="#0f172a")
        footer.pack(fill="x", padx=28, pady=(14, 22))
        tk.Label(
            footer,
            textvariable=self.status_var,
            font=("Microsoft YaHei UI", 10),
            fg="#cbd5e1",
            bg="#0f172a",
            anchor="w",
        ).pack(fill="x")

    def _section_title(self, parent: tk.Frame, text: str) -> None:
        tk.Label(
            parent,
            text=text,
            font=("Microsoft YaHei UI", 14, "bold"),
            fg="#e2e8f0",
            bg="#111827",
        ).pack(anchor="w", padx=18, pady=(18, 10))

    def _button(self, parent: tk.Frame, text: str, command, primary: bool = False) -> None:
        bg = "#0ea5e9" if primary else "#1f2937"
        active_bg = "#0284c7" if primary else "#334155"
        button = tk.Button(
            parent,
            text=text,
            command=command,
            font=("Microsoft YaHei UI", 11),
            fg="#f8fafc",
            bg=bg,
            activeforeground="#ffffff",
            activebackground=active_bg,
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=14,
            pady=11,
        )
        button.pack(fill="x", padx=18, pady=6)

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def start_demo(self) -> None:
        self.start_api()
        self.after(2000, self.start_web)
        self.after(5500, self.open_web)
        self.set_status("已按顺序启动后端和前端；如果端口被占用，请查看弹出的命令窗口。")

    def start_api(self) -> None:
        start_shell("CyberDD API Backend", command_line([PYTHON, "api.py"]))
        self.set_status("已启动 API 后端窗口：http://localhost:8000")

    def start_web(self) -> None:
        if not (WEB_DIR / "package.json").exists():
            messagebox.showerror("前端目录缺失", "没有找到 web/package.json。")
            return
        start_shell("CyberDD Web Frontend", "pnpm run dev", WEB_DIR)
        self.set_status("已启动 Web 前端窗口：http://localhost:5173")

    def open_web(self) -> None:
        webbrowser.open("http://localhost:5173")
        self.set_status("已打开 WebUI：http://localhost:5173")

    def open_api_docs(self) -> None:
        webbrowser.open("http://localhost:8000/docs")
        self.set_status("已打开 API 文档：http://localhost:8000/docs")

    def run_checks(self) -> None:
        start_shell("CyberDD Quality Checks", command_line([PYTHON, "tools/run_all_checks.py"]))
        self.set_status("已启动完整质量检查。")

    def build_demo_system(self) -> None:
        start_shell("CyberDD Build Demo System", command_line([PYTHON, "tools/build_demo_system.py"]))
        self.set_status("已启动演示系统刷新任务。")

    def generate_report(self) -> None:
        report_command = " && ".join(
            [
                command_line([PYTHON, "tools/generate_model_manifest.py"]),
                command_line([PYTHON, "tools/generate_project_report.py"]),
            ]
        )
        start_shell("CyberDD Project Report", report_command)
        self.set_status("已启动项目报告生成任务：outputs/project_report.md")

    def package_release(self) -> None:
        start_shell("CyberDD Package Release", command_line([PYTHON, "tools/package_release.py"]))
        self.set_status("已启动发布包生成任务：release/cyberdd_release.zip")

    def open_project_dir(self) -> None:
        if os.name == "nt":
            os.startfile(PROJECT_DIR)  # type: ignore[attr-defined]
        else:
            webbrowser.open(PROJECT_DIR.as_uri())
        self.set_status("已打开项目目录。")

    def open_cmd_menu(self) -> None:
        start_shell("CyberDD Command Menu", "run.cmd")
        self.set_status("已打开命令行菜单 run.cmd。")


if __name__ == "__main__":
    app = CyberDDLauncher()
    app.mainloop()
