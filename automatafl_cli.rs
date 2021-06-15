mod automatafl;
use automatafl::*;

use std::str::FromStr;
use std::num::ParseIntError;
use std::fmt::{self, Display, Formatter};

#[derive(Debug,Clone)]
pub struct State {
    pub game: Automatafl,
}

#[derive(Debug,Clone,Copy,PartialEq,Eq,Hash)]
pub enum Command {
    Reset,
    Propose(Player, PlayerMove),
    Put(Pos, Option<Piece>),
    AgentMove,
    Help,
    Exit,
}

#[derive(Debug,Clone,PartialEq,Eq)]
pub enum CommandParseError {
    NoCommand,
    UnknownCommand(String),
    ParseIntError(ParseIntError),
    TooFewArguments(usize),
    UnknownColor(String),
}

impl From<ParseIntError> for CommandParseError {
    fn from(i: ParseIntError) -> Self {
        CommandParseError::ParseIntError(i)
    }
}

impl FromStr for Command {
    type Err = CommandParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use Command::*;
        use CommandParseError::*;

        let words: Vec<&str> = s.split_whitespace().collect();
        if let Some(&cmd) = words.first() {
            let args = &words[1..];
            match cmd {
                "reset" | "r" => Ok(Reset),
                "propose" | "p" => {
                    let arity = || TooFewArguments(4);
                    let (ply, fx, fy, tx, ty) = (
                        args.get(0).map(|s| s.parse::<u32>()).ok_or_else(arity)??,
                        args.get(1).map(|s| s.parse::<usize>()).ok_or_else(arity)??,
                        args.get(2).map(|s| s.parse::<usize>()).ok_or_else(arity)??,
                        args.get(3).map(|s| s.parse::<usize>()).ok_or_else(arity)??,
                        args.get(4).map(|s| s.parse::<usize>()).ok_or_else(arity)??,
                    );
                    Ok(Propose(ply as Player, PlayerMove { from: (fx, fy), to: (tx, ty) }))
                },
                "amove" | "a" => Ok(AgentMove),
                "put" => {
                    let arity = || TooFewArguments(3);
                    let (x, y, &p) = (
                        args.get(0).map(|s| s.parse::<usize>()).ok_or_else(arity)??,
                        args.get(1).map(|s| s.parse::<usize>()).ok_or_else(arity)??,
                        args.get(2).ok_or_else(arity)?,
                    );
                    let piece = match p {
                        "A" => Some(Piece::Color(Color::Attractor)),
                        "R" => Some(Piece::Color(Color::Repulsor)),
                        "a" => Some(Piece::Agent),
                        "_" => None,
                        s => return Err(UnknownColor(s.to_string())),
                    };
                    Ok(Put((x, y), piece))
                },
                "help" | "h" | "?" => Ok(Help),
                "exit" | "e" | "quit" | "q" => Ok(Exit),
                s => Err(UnknownCommand(s.to_string())),
            }
        } else {
            Err(NoCommand)
        }
    }
}

impl State {
    pub fn run_command(&mut self, cmd: Command) {
        match cmd {
            Command::Reset => {
                self.game = Automatafl::new_harding_arlynx_11_11(Players::Two);
            },
            Command::Propose(ply, mv) => {
                self.game.propose(ply, mv);
            },
            Command::Put(pos, piece) => {
                self.game.place(pos, piece);
            },
            Command::AgentMove => {
                self.game.agent_move();
            },
            Command::Help => {
                eprintln!("Commands:
- [r]eset: return the board to initial state
- [p]ropose <player> <fx> <fy> <tx> <ty>: propose move from <player> (0-1 or 0-3), from (<fx>,<fy>) to (<tx>,<ty>)
  (check output events for success state)
- [a]move: force the agent to move right now
- put <x> <y> <piece>: put the piece (A: attractor, R: repulsor, a: agent, _: nothing (remove a piece)) at (x,y)
- [h]elp / ?: show this help
- [e]xit / [q]uit: end this program");
            },
            Command::Exit => {
                std::process::exit(0);
            }
        }
    }
}

impl Display for State {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.game.board)?;
        let mut cons = Vec::new();
        self.game.orthoconsiderations(&mut cons);
        let mut any = false;
        for (pos, oc) in cons.drain(..) {
            any = true;
            write!(f, "Agent @{:?}: Consideration: {:?}\nOn X: {:?}\nOn Y: {:?}", pos, oc, oc.x.to_move(), oc.y.to_move())?;
            if let Some(dir) = oc.best_move() {
                write!(f, "\nMove: {:?} ({:?})", dir, dir.to_move(pos))?;
            } else {
                write!(f, "\nNo move.")?;
            }
        }
        if !any {
            write!(f, "No considerations.")?;
        }
        Ok(())
    }
}

fn main() {
    let mut state = State { game: Automatafl::new_harding_arlynx_11_11(Players::Two) };
    let mut line = String::new();
    let stdin = std::io::stdin();

    eprintln!("{}", state);

    loop {
        line.clear();
        if let Err(e) = stdin.read_line(&mut line) {
            eprintln!("Failed to read ({:?}), exiting", e);
            break;
        }

        match line.trim().parse::<Command>() {
            Err(ce) => eprintln!("Error parsing command: {:?}", ce),
            Ok(c) => {
                state.run_command(c);
                for ev in state.game.pump() {
                    eprintln!("Event: {:?}", ev);
                }
                eprintln!("{}", state);
            },
        }
    }
}
