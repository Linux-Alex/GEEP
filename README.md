# GEEP (Genetic Evolutionary Engineering Platform)

GEEP (Genetic Evolutionary Engineering Platform) is a powerful **C++ framework** for genetic programming, designed to enable the evolution of complex solutions through large-scale population-based optimization. Leveraging **CUDA acceleration**, GEEP parallelizes fitness evaluation and genetic operations, making it highly efficient for handling massive populations. This framework is ideal for researchers and developers aiming to solve high-dimensional problems using evolutionary algorithms while harnessing the power of GPU computing.

## 🚧 Project Status
**GEEP is currently in its early development phase.**

We are actively working on building the core functionalities, including classes, relationships, and foundational algorithms. At this stage, we are also testing these components through various examples to ensure robustness and correctness. Key features like GEEPlang and the web dashboard are planned for future development.

## 🚀 Key Features

- **Genetic Programming Framework:** Evolve solutions for complex problems using population-based optimization.
- **CUDA Acceleration:** Parallelize fitness evaluation and genetic operations for high-performance computing.
- **Large-Scale Populations:** Efficiently handle massive populations for better optimization results.
- **C++ with Qt:** A user-friendly interface for seamless development and experimentation.
- **Inspired by EARS:** Built with inspiration from the [EARS framework](https://github.com/UM-LPM/EARS).
- **Web Service:** Operates as a web service for remote execution of genetic programs.

## 🛠️ Getting Started

**Prerequisites:**
- **CUDA Toolkit:** Ensure CUDA is installed on your system.
- **C++ Compiler:** A modern C++ compiler (e.g., GCC, Clang, or MSVC).
- **Optional:** Mandatory for the graphical interface and core functionality.

**Installation:**

1. Clone the repository:
```
git clone https://github.com/Linux-Alex/GEEP.git
```
2. Build the project:
```
cd GEEP
mkdir build && cd build
cmake ..
make
```
3. Run the example:
```
./GEEP --example p:001 --check-cuda
```

## 📖 Usage
Run the GEEP framework with the following command-line options:
```
$ ./GEEP -h
Usage: ./GEEP [options]
GEEP Framework - Genetic Evolutionary Engineering Platform

Options:
  -h, --help                Displays help on commandline options.
  --help-all                Displays help, including generic Qt options.
  -v, --version             Displays version information.
  --service                 Start as a service (runs forever).
  --input <file>            Input XML file.
  --output-dir <directory>  Specify the output directory.
  --server-port <port>      Specify the port number.
  --example <program>       Run an example program.
  --example-list <type>     List examples by type: all|program|xml.
  --check-cuda              Check for CUDA support.
```

## 🧑‍💻 About Me

I am a master's student at the University of Maribor, Faculty of Electrical Engineering and Computer Science (UM FERI), developing GEEP as part of my master's thesis. This project has been a passion of mine for over two years, with the current inspiration coming from EARS, a Java-based framework for ranking and experimenting with evolutionary algorithms, developed by the LPM Laboratory at UM FERI.

## 📄 Project Origin and Motivation

The idea for GEEP originated two years ago during my studies at FERI, in the course *Introduction to Evolutionary Algorithms*. Genetic programming fascinated me because it offers greater control over the optimization process compared to neural networks and allows for the seamless transfer of natural concepts into algorithms. This approach is particularly useful for scientists from other fields who seek efficient and understandable methods for modeling complex problems.

The name **GEEP** is inspired by the biological phenomenon of a **sheep-goat hybrid**, known for its rarity and unique characteristics. Analogously, GEEP combines diverse approaches to genetic programming into a unified, powerful framework ([logo idea](assets/logo.png)).

## 📄 License

This project is licensed under the **MIT License**.

## 📧 Contact

For questions or collaborations, feel free to reach out:
- Email: aleks.marinic@kdemail.net
- GitHub: [Linux-Alex](https://github.com/linux-Alex/)
