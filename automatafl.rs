use std::{iter, ops};
use std::cmp::{Ord, PartialOrd, Ordering};
use std::fmt::{self, Display, Formatter};
use std::collections::{HashSet, HashMap};

#[derive(Debug,Clone,Copy,PartialEq,Eq,Hash)]
pub enum Color {
    Attractor,
    Repulsor,
}

impl Display for Color {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use Color::*;
        match self {
            Attractor => write!(f, "A"),
            Repulsor => write!(f, "R"),
        }
    }
}

#[derive(Debug,Clone,Copy,PartialEq,Eq,Hash)]
pub enum Piece {
    Color(Color),
    Agent,
}

impl Display for Piece {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Piece::Color(c) => write!(f, "{}", c),
            Piece::Agent => write!(f, "a"),
        }
    }
}

pub type Player = u32;
pub type Pos = (usize, usize);
pub type SPos = (isize, isize);

#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub struct Cell {
    pub piece: Option<Piece>,
    pub conflict: bool,
    pub passable: bool,
    pub goal: Option<Player>,
}

impl Display for Cell {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}{}{}",
               if let Some(py) = self.goal { format!("{}", py) } else { " ".to_string() },
               self.piece.map(|p| format!("{}", p)).unwrap_or_else(|| "_".to_string()),
               match (self.conflict, self.passable) {
                   (false, false) => " ",
                   (true, false) => "!",
                   (false, true) => "P",
                   (true, true) => "?",
               },
        )
    }
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            piece: None,
            conflict: false,
            passable: false,
            goal: None,
        }
    }
}

#[derive(Debug,Clone,Copy,PartialEq,Eq,Hash)]
pub enum Dir {
    PX, NX, PY, NY,
}

#[derive(Debug,Clone,Copy,PartialEq,Eq,Hash)]
pub enum Axis {
    X, Y
}

impl Dir {
    pub fn get_offset(self) -> SPos {
        use Dir::*;
        match self {
            PX => (1, 0),
            NX => (-1, 0),
            PY => (0, 1),
            NY => (0, -1),
        }
    }
    
    pub fn offset(self, pos: Pos) -> Pos {
        use Dir::*;
        match self {
            PX => (pos.0.saturating_add(1), pos.1),
            NX => (pos.0.saturating_sub(1), pos.1),
            PY => (pos.0, pos.1.saturating_add(1)),
            NY => (pos.0, pos.1.saturating_sub(1)),
        }
    }

    pub fn to_move(self, pos: Pos) -> PlayerMove {
        PlayerMove {
            from: pos,
            to: self.offset(pos),
        }
    }

    pub fn axis(self) -> Axis {
        use Dir::*;
        match self {
            PX | NX => Axis::X,
            PY | NY => Axis::Y,
        }
    }

    pub fn is_negative(self) -> bool {
        use Dir::*;
        match self {
            PX | PY => false,
            NX | NY => true,
        }
    }
}

impl Axis {
    pub fn select(self, pos: Pos) -> usize {
        use Axis::*;
        match self {
            X => pos.0,
            Y => pos.1,
        }
    }
}

#[derive(Debug,Clone)]
pub struct Board<C> {
    storage: Vec<C>,
    width: usize,
    height: usize,
}

pub struct BoardRay<'a, C> {
    board: &'a Board<C>,
    current: Pos,
    dir: Dir,
    limit: usize,
}

impl<C> Board<C> {
    pub fn new_with<F>(width: usize, height: usize, gen: F) -> Self
    where F: FnMut() -> C
    {
        Self {
            storage: iter::repeat_with(gen).take(width * height).collect(),
            width, height,
        }
    }

    pub fn new(width: usize, height: usize) -> Self
    where C: Default
    {
        Self::new_with(width, height, Default::default)
    }

    pub fn width(&self) -> usize { self.width }
    pub fn height(&self) -> usize { self.height }
    pub fn size(&self) -> Pos { (self.width, self.height) }

    pub fn clip(&self, pos: Pos) -> Pos {
        (pos.0.min(self.width), pos.1.min(self.height))
    }
    pub fn contains(&self, pos: Pos) -> bool {
        pos.0 < self.width && pos.1 < self.height
    }

    pub fn ray<'a>(&'a self, start: Pos, dir: Dir, limit: Option<usize>) -> BoardRay<'a, C> {
        BoardRay {
            board: &self,
            current: start,
            dir,
            limit: limit.unwrap_or_else(|| {
                if dir.is_negative() {
                    0
                } else {
                    dir.axis().select(self.size()) - 1
                }
            }),
        }
    }

    fn linear_index(&self, pos: Pos) -> usize {
        pos.1 * self.width + pos.0
    }
}

impl<C> ops::Index<Pos> for Board<C> {
    type Output = C;
    fn index(&self, idx: Pos) -> &Self::Output {
        if !self.contains(idx) {
            panic!("Board index {:?} out of bounds {:?}", idx, self.size());
        }
        let idx = self.linear_index(idx);
        &self.storage[idx]
    }
}

impl<C> ops::IndexMut<Pos> for Board<C> {
    fn index_mut(&mut self, idx: Pos) -> &mut Self::Output {
        if !self.contains(idx) {
            panic!("Board index {:?} out of bounds {:?}", idx, self.size());
        }
        let idx = self.linear_index(idx);
        &mut self.storage[idx]
    }
}

impl<C: Display> Display for Board<C> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        // XXX can't map this due to carrier?
        for y in 0..self.height {
            for x in 0..self.width {
                write!(f, "{} ", self[(x, y)])?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}


impl<'a, C> Iterator for BoardRay<'a, C> {
    type Item = (Pos, &'a C);

    fn next(&mut self) -> Option<Self::Item> {
        let onax = self.dir.axis().select(self.current);
        if self.dir.is_negative() {
            if onax <= self.limit {
                return None;
            }
        } else {
            if onax >= self.limit {
                return None;
            }
        }
        self.current = self.dir.offset(self.current);
        Some((self.current, &self.board[self.current]))
    }
}

#[derive(Debug,Clone,Copy,PartialEq,Eq,Hash)]
pub struct PlayerMove {
    pub from: Pos,
    pub to: Pos,
}

#[derive(Debug,Clone,Copy,PartialEq,Eq,Hash)]
pub enum MoveError {
    NonOrthogonal,
    Trivial,
    TargetsAgent,
    TargetsConflict,
    OutOfBounds,
    NoSource,
    DestinationOccupied,
    PathOccupied,
    PresentlyImmutable,
    AlreadyWon,
}

impl MoveError {
    pub fn can_reconsider(self) -> bool {
        use MoveError::*;
        match self {
            NoSource | DestinationOccupied => true,
            _ => false,
        }
    }
}

impl PlayerMove {
    pub fn is_ortho(self) -> bool {
        self.from.0 == self.to.0 || self.from.1 == self.to.1
    }

    pub fn dir(self) -> Option<Dir> {  // Only accurate if is_ortho
        use self::Ordering::*;
        use Dir::*;
        match (self.to.0.cmp(&self.from.0), self.to.1.cmp(&self.from.1)) {
            (Greater, _) => Some(PX),
            (Less, _) => Some(NX),
            (_, Greater) => Some(PY),
            (_, Less) => Some(NY),
            _ => None,
        }
    }
}

#[derive(Debug,Clone)]
pub struct Automatafl {
    pub board: Board<Cell>,
    agents: HashSet<Pos>,
    players: Players,
    players_set: HashSet<Player>,
    locked: HashSet<Player>,
    moves: Vec<Option<PlayerMove>>,  // Indexed by Player
    event_queue: Vec<Event>,
    winners: HashSet<Player>,
    conflict_positions: Vec<Pos>,
}

macro_rules! spec_piece {
    (A) => {Some(Piece::Color(Color::Attractor))};
    (R) => {Some(Piece::Color(Color::Repulsor))};
    (_) => {None};
}

macro_rules! piece_board {
    [ $( [ $( $spec:tt ),* ] ),* ] => {{
        let converted = vec![ $( vec![ $( spec_piece!($spec), )* ], )* ];
        let height = converted.len();
        let width = converted.first().unwrap().len();
        let mut board: Board<Cell> = Board::new(width, height);
        for (row_pos, row) in converted.into_iter().enumerate() {
            for (col_pos, piece) in row.into_iter().enumerate() {
                if let Some(pc) = piece {
                    board[(col_pos, row_pos)].piece = Some(pc);
                }
            }
        }
        board
    }};
}

#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub enum Players {
    Two,
    Four,
}

impl Players {
    pub const fn count(self) -> usize {
        use Players::*;
        match self {
            Two => 2,
            Four => 4,
        }
    }

    pub fn set(self) -> HashSet<Player> {
        (0 .. (self.count() as Player)).into_iter().collect()
    }
}

#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub struct Placement(Color, usize);

#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub struct Consideration {
    pub pos: Option<Placement>,
    pub neg: Option<Placement>,
    pub loc: usize,
    pub size: usize,
}

#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub struct OrthoConsideration {
    pub x: Consideration,
    pub y: Consideration,
}

#[derive(Debug,Clone)]
pub enum EventData {
    MoveAcknowledged { player: Player, mv: PlayerMove, },
    MoveReady { player: Player, },
    MoveIllegal { player: Player, mv: PlayerMove, err: MoveError, },
    MoveInvalid { player: Player, mv: PlayerMove, err: MoveError, },
    Conflict { players: Vec<Player>, moves: Vec<PlayerMove>, pos: Pos, },  // The vecs are parallel
    Moved { player: Player, mv: PlayerMove, },
    AgentMove { agent_move: PlayerMove, },
    TurnOver { winners: Vec<Player> },
}

#[derive(Debug,Clone)]
pub struct Event {
    data: EventData,
    dest: Player,
}

impl Automatafl {
    pub fn new_harding_arlynx_11_11(ply: Players) -> Self {
        let mut instance = Self {
            board: piece_board![
                [R, _, _, _, A, R, A, _, _, _, R],
                [R, _, _, _, A, R, A, _, _, _, R],
                [_, _, _, _, _, _, _, _, _, _, _],
                [_, A, _, _, _, _, _, _, _, A, _],
                [R, R, _, _, _, _, _, _, _, R, R],
                [R, R, _, _, _, _, _, _, _, R, R],
                [R, R, _, _, _, _, _, _, _, R, R],
                [_, A, _, _, _, _, _, _, _, A, _],
                [_, _, _, _, _, _, _, _, _, _, _],
                [R, _, _, _, A, R, A, _, _, _, R],
                [R, _, _, _, A, R, A, _, _, _, R]
            ],
            agents: HashSet::new(),
            players: ply,
            players_set: ply.set(),
            locked: HashSet::new(),
            moves: iter::repeat(None).take(ply.count()).collect(),
            event_queue: Vec::new(),
            winners: HashSet::new(),
            conflict_positions: Vec::new(),
        };
        instance.place((5, 5), Some(Piece::Agent));
        match ply {
            Players::Two => {
                instance.board[(0, 0)].goal = Some(0);
                instance.board[(10, 0)].goal = Some(0);
                instance.board[(0, 10)].goal = Some(1);
                instance.board[(10, 10)].goal = Some(1);
            },
            Players::Four => {
                // FIXME
                eprintln!("Automatafl::new_harding_arlynx_11_11(Players::Four) is unattested and not standard");
                instance.board[(0, 0)].goal = Some(0);
                instance.board[(0, 10)].goal = Some(1);
                instance.board[(10, 0)].goal = Some(2);
                instance.board[(10, 10)].goal = Some(3);
            },
        }
        instance
    }

    pub fn new_grissess_5_5(ply: Players) -> Self {
        eprintln!("Automatafl::new_grissess_5_5(_) is unattested and not standard");
        let mut instance = Self {
            board: piece_board![
                [R, _, A, _, R],
                [_, _, _, _, _],
                [R, _, _, _, R],
                [_, _, _, _, _],
                [R, _, A, _, R]
            ],
            agents: HashSet::new(),
            players: ply,
            players_set: ply.set(),
            locked: HashSet::new(),
            moves: iter::repeat(None).take(ply.count()).collect(),
            event_queue: Vec::new(),
            winners: HashSet::new(),
            conflict_positions: Vec::new(),
        };
        instance.place((2, 2), Some(Piece::Agent));
        match ply {
            Players::Two => {
                instance.board[(0, 0)].goal = Some(0);
                instance.board[(4, 0)].goal = Some(0);
                instance.board[(0, 4)].goal = Some(1);
                instance.board[(4, 4)].goal = Some(1);
            },
            Players::Four => {
                instance.board[(0, 0)].goal = Some(0);
                instance.board[(0, 4)].goal = Some(1);
                instance.board[(4, 0)].goal = Some(2);
                instance.board[(4, 4)].goal = Some(3);
            },
        }
        instance
    }

    pub fn move_dir(&mut self, pos: Pos, dir: Dir) {
        self.exec_move(
            dir.to_move(pos),
        );
    }

    pub fn num_agents(&self) -> usize { self.agents.len() }

    pub fn exec_move(&mut self, mv: PlayerMove) {
        if mv.from == mv.to {
            return;
        }

        match (
            self.agents.contains(&mv.from),
            self.agents.contains(&mv.to),
        ) {
            (false, false) | (true, true) => (),
            (true, false) => {  // From -> To
                self.agents.remove(&mv.from);
                self.agents.insert(mv.to);
            },
            (false, true) => {  // To -> From (dubious)
                self.agents.remove(&mv.to);
                self.agents.insert(mv.from);
            }
        }

        unsafe {
            let pp_from = (&mut self.board[mv.from].piece) as *mut Option<Piece>;
            let pp_to = (&mut self.board[mv.to].piece) as *mut Option<Piece>;
            // Safety: The pointers are read/write safe as they're generated from safe references
            // (and, indeed, the borrow is split on the precondition above).
            pp_from.swap(pp_to);
        }
    }

    pub fn place(&mut self, pos: Pos, piece: Option<Piece>) {
        self.agents.remove(&pos);
        self.board[pos].piece = piece;
        if let Some(Piece::Agent) = piece {
            self.agents.insert(pos);
        }
    }

    pub fn add_conflict(&mut self, pos: Pos) {
        self.board[pos].conflict = true;
        self.conflict_positions.push(pos);
    }

    pub fn reset_conflicts(&mut self) {
        for pos in self.conflict_positions.drain(..) {
            self.board[pos].conflict = false;
        }
    }

    pub fn orthoconsiderations(&self, into: &mut Vec<(Pos, OrthoConsideration)>) {
        into.clear();
        into.reserve(self.num_agents());

        into.extend(self.agents.iter().map(|&pos| {
            (pos, OrthoConsideration {
                x: Consideration {
                    neg: self.placement_from(pos, Dir::NX),
                    pos: self.placement_from(pos, Dir::PX),
                    loc: pos.0,
                    size: self.board.width(),
                },
                y: Consideration {
                    neg: self.placement_from(pos, Dir::NY),
                    pos: self.placement_from(pos, Dir::PY),
                    loc: pos.1,
                    size: self.board.height(),
                },
            })
        }));
    }

    pub fn placement_from(&self, pos: Pos, dir: Dir) -> Option<Placement> {
        for (dist, (_p, cell)) in self.board.ray(pos, dir, None).enumerate() {
            if let Some(Piece::Color(col)) = cell.piece {
                return Some(Placement(col, dist + 1));
            }
        }
        None
    }

    pub fn move_legal(&self, mv: PlayerMove) -> Result<(), MoveError> {
        use MoveError::*;
        if !self.winners.is_empty() { return Err(AlreadyWon); }
        if !mv.is_ortho() { return Err(NonOrthogonal); }
        if mv.from == mv.to { return Err(Trivial); }
        for pos in (&[mv.from, mv.to]).into_iter().cloned() {
            if !self.board.contains(pos) { return Err(OutOfBounds); }
            if self.agents.contains(&pos) { return Err(TargetsAgent); }
            if self.board[pos].conflict { return Err(TargetsConflict); }
        }
        Ok(())
    }

    pub fn move_valid(&self, mv: PlayerMove) -> Result<(), MoveError> {
        use MoveError::*;
        if let Err(e) = self.move_legal(mv) { return Err(e); }
        if let Some(Piece::Color(_)) = self.board[mv.from].piece {
            ()
        } else { return Err(NoSource); }
        if !self.board[mv.to].piece.is_none() {
            return Err(DestinationOccupied);
        }
        let dir = mv.dir().unwrap();
        for (_p, cell) in self.board.ray(mv.from, dir, Some(dir.axis().select(mv.to))) {
            if !cell.passable && cell.piece.is_some() {
                return Err(PathOccupied);
            }
        }
        Ok(())
    }

    fn enqueue(&mut self, ev: Event) {
        self.event_queue.push(ev);
    }

    fn broadcast(&mut self, evd: EventData) {
        for ply in self.players_iter() {
            self.enqueue(Event { data: evd.clone(), dest: ply });
        }
    }

    pub fn players(&self) -> Players { self.players }
    pub fn all_players(&self) -> &HashSet<Player> { &self.players_set }
    pub fn players_iter(&self) -> impl Iterator<Item=Player> {
        (0 .. self.players.count()).map(|p| p as Player)
    }

    pub fn players_ready(&self) -> HashSet<Player> {
        self.moves.iter().cloned().enumerate()
            .filter(|(_p, omv)| omv.is_some())
            .map(|(p, _omv)| p as Player)
            .collect()
    }

    pub fn move_legal_for_player(&self, ply: Player, mv: PlayerMove) -> Result<(), MoveError> {
        use MoveError::*;
        if self.locked.contains(&ply) {
            return Err(PresentlyImmutable);
        }
        self.move_legal(mv)?;
        Ok(())
    }

    pub fn propose(&mut self, ply: Player, mv: PlayerMove) {
        use EventData::*;
        match self.move_legal_for_player(ply, mv) {
            Ok(_) => {
                self.moves[ply as usize] = Some(mv);
                self.broadcast(MoveReady { player: ply, });
                self.enqueue(Event { data: MoveAcknowledged { player: ply, mv, }, dest: ply });
            },
            Err(err) => {
                self.enqueue(Event { data: MoveIllegal { player: ply, mv, err, }, dest: ply });
            },
        }
    }
    
    fn resolve(&mut self) {
        use EventData::*;

        let mut moves: HashMap<Player, PlayerMove> = self.moves.iter().enumerate()
            .filter(|(_p, omv)| omv.is_some())
            .map(|(p, omv)| (p as Player, omv.unwrap()))
            .collect();
        let mut seen_sources: HashMap<Pos, Player> = HashMap::new();
        let mut seen_dests: HashMap<Pos, Player> = HashMap::new();
        let mut seen_moves: HashSet<PlayerMove> = HashSet::new();
        let mut conflicts: HashMap<Pos, HashSet<Player>> = HashMap::new();

        for (&ply, &mv) in &moves {
            if seen_moves.contains(&mv) {
                continue;
            }
            seen_moves.insert(mv);

            if seen_sources.contains_key(&mv.from) {
                let conf_set = conflicts.entry(mv.from).or_insert_with(HashSet::new);
                conf_set.insert(ply);
                conf_set.insert(seen_sources[&mv.from]);
            } else {
                seen_sources.insert(mv.from, ply);
            }

            if seen_dests.contains_key(&mv.to) {
                let conf_set = conflicts.entry(mv.to).or_insert_with(HashSet::new);
                conf_set.insert(ply);
                conf_set.insert(seen_dests[&mv.to]);
            } else {
                seen_dests.insert(mv.to, ply);
            }
        }

        if !conflicts.is_empty() {
            for (pos, ply_set) in conflicts {
                let mut players: Vec<Player> = Vec::new();
                let mut mvs: Vec<PlayerMove> = Vec::new();

                for ply in &ply_set {
                    players.push(*ply);
                    mvs.push(moves[ply]);
                    self.moves[(*ply) as usize] = None;
                }
                self.broadcast(Conflict { players, moves: mvs, pos, });
                self.add_conflict(pos);
            }
            self.locked.clear();
            for (p, omv) in self.moves.iter().enumerate() {
                if omv.is_some() {
                    self.locked.insert(p as Player);
                }
            }
            return;
        }

        while !moves.is_empty() {
            let mut changed = false;
            for (&ply, &mv) in &moves {
                match self.move_valid(mv) {
                    Ok(_) => {
                        self.exec_move(mv);
                        self.broadcast(Moved { player: ply, mv, });
                        moves.remove(&ply);
                        changed = true;
                        break;
                    },
                    Err(err) => {
                        if !err.can_reconsider() {
                            self.broadcast(MoveInvalid { player: ply, mv, err, });
                            moves.remove(&ply);
                            changed = true;
                            break;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }

        for (ply, mv) in moves {
            match self.move_valid(mv) {
                Ok(_) => unreachable!(),
                Err(err) => self.broadcast(MoveInvalid { player: ply, mv, err, }),
            }
        }

        self.agent_move();
        self.turn_over();
    }

    pub fn turn_over(&mut self) {
        use EventData::*;

        for &apos in &self.agents {
            if let Some(ply) = self.board[apos].goal {
                self.winners.insert(ply);
            }
        }
        self.broadcast(TurnOver { winners: self.winners.iter().cloned().collect(), });
        self.reset();
    }

    pub fn reset(&mut self) {
        self.locked.clear();
        self.moves.fill(None);
        self.reset_conflicts();
    }

    pub fn agent_move(&mut self) {
        use EventData::*;

        let mut cons = Vec::new();  // TODO: cache
        self.orthoconsiderations(&mut cons);

        for (pos, oc) in cons.drain(..) {
            if let Some(dir) = oc.best_move() {
                let mv = dir.to_move(pos);
                self.exec_move(mv);
                self.broadcast(AgentMove { agent_move: mv, });
            }
        }
    }

    pub fn pump(&mut self) -> Vec<Event> {
        if self.moves.iter().all(Option::is_some) {
            self.resolve();
        }
        self.event_queue.drain(..).collect()
    }
}

#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub enum AgentMoveSign {
    Neg,
    Pos,
    Zero,
}

#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub enum Reason {
    UnbalancedPair { adist: usize, rdist: usize },
    FromRepulsor { rdist: usize },
    TowardAttractor { adist: usize },
    NoReason,
}

impl Reason {
    pub fn discrim_priority(self) -> usize {
        use Reason::*;
        match self {
            UnbalancedPair {..} => 3,
            FromRepulsor {..} => 2,
            TowardAttractor {..} => 1,
            NoReason => 0,
        }
    }
}

impl PartialOrd for Reason {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Reason {
    fn cmp(&self, other: &Self) -> Ordering {
        use Reason::*;
        self.discrim_priority().cmp(&other.discrim_priority())
            .then_with(|| {
                match (self, other) {
                    (UnbalancedPair { adist: sa, rdist: sr }, UnbalancedPair {adist: oa, rdist: or}) => {
                        if sa > oa {
                            Ordering::Greater
                        } else if sa < oa {
                            Ordering::Less
                        } else if sr < or {
                            Ordering::Greater
                        } else if sr > or {
                            Ordering::Less
                        } else {
                            Ordering::Equal
                        }
                    },
                    (FromRepulsor { rdist: sr }, FromRepulsor { rdist: or }) => or.cmp(sr),
                    (TowardAttractor { adist: sa }, TowardAttractor { adist: oa}) => oa.cmp(sa),
                    (NoReason, NoReason) => Ordering::Equal,
                    _ => unreachable!(),
                }
            })
    }
}

#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub struct AgentMove {
    pub reason: Reason,
    pub dir: AgentMoveSign,
}

impl Consideration {
    pub fn to_move(self) -> AgentMove {
        use Color::*;
        use Reason::*;
        use AgentMoveSign::*;
        match (self.pos, self.neg) {
            (Some(Placement(Attractor, ad)), Some(Placement(Repulsor, rd))) if ad > 1 =>
                AgentMove { reason: UnbalancedPair { adist: ad, rdist: rd }, dir: Pos },
            (Some(Placement(Repulsor, rd)), Some(Placement(Attractor, ad))) if ad > 1 =>
                AgentMove { reason: UnbalancedPair { adist: ad, rdist: rd }, dir: Neg },

            (Some(Placement(Repulsor, rd)), None) if self.loc > 0 =>
                AgentMove { reason: FromRepulsor { rdist: rd }, dir: Neg },
            (None, Some(Placement(Repulsor, rd))) if self.loc < self.size - 1 =>
                AgentMove { reason: FromRepulsor { rdist: rd }, dir: Pos },
            (Some(Placement(Repulsor, prd)), Some(Placement(Repulsor, nrd))) if prd != nrd =>
                AgentMove { reason: FromRepulsor { rdist: prd.min(nrd) }, dir: if prd < nrd { Neg } else { Pos } },

            (Some(Placement(Attractor, ad)), None) if ad > 1 =>
                AgentMove { reason: TowardAttractor { adist: ad }, dir: Pos },
            (None, Some(Placement(Attractor, ad))) if ad > 1 =>
                AgentMove { reason: TowardAttractor { adist: ad }, dir: Neg },
            (Some(Placement(Attractor, pad)), Some(Placement(Attractor, nad))) if pad != nad && pad.min(nad) > 1 =>
                AgentMove { reason: TowardAttractor { adist: pad.min(nad) }, dir: if pad < nad { Pos } else { Neg } },

            _ => AgentMove { reason: NoReason, dir: Zero },
        }
    }
}

impl AgentMoveSign {
    pub fn to_dir(self, axis: Axis) -> Option<Dir> {
        use AgentMoveSign::*;
        use Axis::*;
        use Dir::*;
        match axis {
            X => match self {
                Pos => Some(PX),
                Neg => Some(NX),
                Zero => None,
            },
            Y => match self {
                Pos => Some(PY),
                Neg => Some(NY),
                Zero => None,
            }
        }
    }
}

impl OrthoConsideration {
    pub fn best_move(self) -> Option<Dir> {
        let x = self.x.to_move();
        let y = self.y.to_move();
        if x.reason < y.reason {  // XXX: column rule
            y.dir.to_dir(Axis::Y)
        } else {
            x.dir.to_dir(Axis::X)
        }
    }
}

fn main() {
    let mut game = Automatafl::new_harding_arlynx_11_11(Players::Two);
    println!("{}", game.board);
    let mut conss = Vec::new();
    game.orthoconsiderations(&mut conss);
    let cons = conss.first().unwrap().1;
    println!("Agent movement: {:?} (move {:?})", cons, cons.best_move());
    println!("Components: {:?}, {:?}", cons.x.to_move(), cons.y.to_move());

    //game.place((9, 7), Some(Piece::Color(Color::Repulsor)));
    game.exec_move(PlayerMove { from: (9, 7), to: (5, 7) });

    println!("{}", game.board);
    game.orthoconsiderations(&mut conss);
    let cons = conss.first().unwrap().1;
    println!("Agent movement: {:?} (move {:?})", cons, cons.best_move());
    println!("Components: {:?}, {:?}", cons.x.to_move(), cons.y.to_move());
}
