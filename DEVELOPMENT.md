# Development & AI Context

Looking for instructions on how BeatSync SO-101 is designed, built, and extended?

- 📖 **Architecture & System Design**: See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for data flow, state diagrams, and module breakdowns.
- 🤖 **Developer & Model Context Guide**: See [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) for safety invariants, how to add dance poses, hardware setup, and extension recipes (ElevenLabs, streaming audio, computer vision).
- 🏆 **Hackathon Details**: See [docs/HACK-INFO.md](docs/HACK-INFO.md) for schedule, judging criteria, and venue info.

---

### Quick Verification
To verify the system end-to-end in simulation mode (no hardware required):
```bash
python main.py --demo --sim
```
