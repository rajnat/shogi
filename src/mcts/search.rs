/// MCTS search phases: selection, expansion, evaluation, backpropagation.
use std::sync::{Arc, Mutex};
use rand::Rng;
use rand::distributions::WeightedIndex;
use rand::seq::SliceRandom;
use rand_distr::{Distribution, Gamma};
use crate::board::Board;
use crate::movegen::generate_legal_moves;
use crate::moves::{make_move_full, unmake_move_full, UndoState};
use crate::types::{Color, Move};
use super::{Arena, MctsConfig, Node, NodeIdx, NO_PARENT};

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------

/// Walk from `root` down the tree using PUCT until an unexpanded leaf is
/// reached, applying each move to `board` along the way.
///
/// Returns `(leaf, undo_stack)`.  The caller is responsible for reversing
/// `undo_stack` to restore the board to its state before `select` was called.
pub fn select(
    arena: &Arena,
    root: NodeIdx,
    board: &mut Board,
    c_puct: f32,
) -> (NodeIdx, Vec<(Move, UndoState)>) {
    let mut node_idx = root;
    let mut undo_stack: Vec<(Move, UndoState)> = Vec::new();

    loop {
        if arena.get(node_idx).is_leaf() {
            break;
        }
        // best_child returns None only for leaves, already guarded above.
        let child_idx = arena
            .best_child(node_idx, c_puct)
            .expect("non-leaf must have a best child");

        let mv = arena
            .get(child_idx)
            .mv
            .expect("every non-root node must carry the move that created it");

        let undo = make_move_full(board, mv);
        undo_stack.push((mv, undo));
        node_idx = child_idx;
    }

    (node_idx, undo_stack)
}

// ---------------------------------------------------------------------------
// Expansion
// ---------------------------------------------------------------------------

/// Expand `leaf` by generating all legal moves from `board` and allocating
/// one child node per move with uniform prior probabilities.
///
/// Returns `true` if at least one child was created (non-terminal position),
/// or if the node was already expanded by another thread.
/// Returns `false` if there are no legal moves (checkmate / stalemate).
///
/// The already-expanded guard prevents duplicate children when two parallel
/// threads both select the same leaf before either has expanded it.
pub fn expand(arena: &mut Arena, leaf: NodeIdx, board: &mut Board) -> bool {
    if !arena.get(leaf).is_leaf() {
        return true; // already expanded by another thread
    }

    let mut moves = Vec::new();
    generate_legal_moves(board, &mut moves);

    if moves.is_empty() {
        return false;
    }

    let prior = 1.0 / moves.len() as f32;

    for mv in moves {
        let child_idx = arena.alloc(Node::new(Some(mv), prior, leaf));
        arena.get_mut(leaf).children.push(child_idx);
    }

    true
}

// ---------------------------------------------------------------------------
// Evaluation — random rollout (uniform policy placeholder)
// ---------------------------------------------------------------------------

/// Default cap on rollout depth. Shogi games rarely exceed 300 moves;
/// 200 is enough to reach a terminal in the vast majority of rollouts.
pub const DEFAULT_ROLLOUT_DEPTH: usize = 200;

/// Play random moves from `board` until checkmate or `max_depth` is reached.
///
/// Returns the outcome **from the perspective of the side to move at the time
/// of the call**:
///   +1.0  — that side wins
///   −1.0  — that side loses
///    0.0  — draw (max depth reached without a terminal)
///
/// The board passed in is never modified; rollout works on an internal clone.
/// This function is the placeholder until the neural network (M5) takes over.
pub fn rollout<R: Rng>(board: &Board, rng: &mut R, max_depth: usize) -> f32 {
    let mut b = board.clone();
    let initial_side: Color = b.side_to_move;

    for _ in 0..max_depth {
        let mut moves = Vec::new();
        generate_legal_moves(&mut b, &mut moves);

        if moves.is_empty() {
            // The side currently to move has no legal moves — they lose.
            return if b.side_to_move == initial_side { -1.0 } else { 1.0 };
        }

        let &mv = moves.choose(rng).expect("moves is non-empty");
        let _ = make_move_full(&mut b, mv);
    }

    0.0 // max depth reached — treat as draw
}

// ---------------------------------------------------------------------------
// Backpropagation
// ---------------------------------------------------------------------------

/// Walk from `leaf` up to the root, incrementing visit counts and accumulating
/// values with alternating sign.
///
/// Each node stores value from the perspective of the **side to move at that
/// node**.  Because consecutive nodes alternate sides, the value must be
/// negated at every step going up the tree.
///
/// `value` should be in [−1, 1] from the perspective of the side to move at
/// `leaf` (+1 = that side wins, −1 = that side loses).
pub fn backprop(arena: &mut Arena, leaf: NodeIdx, value: f32) {
    let mut node_idx = leaf;
    let mut v = value;

    loop {
        let node = arena.get_mut(node_idx);
        node.visit_count += 1;
        node.total_value += v;

        let parent = node.parent;
        if parent == super::NO_PARENT {
            break;
        }
        v = -v;
        node_idx = parent;
    }
}

// ---------------------------------------------------------------------------
// Dirichlet noise
// ---------------------------------------------------------------------------

/// Mix Dirichlet(α, …, α) noise into the prior probabilities of `root`'s
/// children, encouraging exploration during self-play.
///
/// Sampling recipe: draw n independent Gamma(α, 1) values, normalize to
/// sum 1 to obtain η ~ Dirichlet(α, …, α), then blend:
///   P'(a) = (1 − ε) · P(a) + ε · η(a)
///
/// Must be called **after** the root has been expanded.
/// Has no effect if the root has no children.
pub fn add_dirichlet_noise<R: Rng>(
    arena: &mut Arena,
    root: NodeIdx,
    alpha: f32,
    epsilon: f32,
    rng: &mut R,
) {
    let n = arena.get(root).children.len();
    if n == 0 {
        return;
    }

    let gamma = Gamma::new(alpha as f64, 1.0).expect("dirichlet_alpha must be > 0");
    let mut noise: Vec<f32> = (0..n).map(|_| gamma.sample(rng) as f32).collect();
    let sum: f32 = noise.iter().sum();
    if sum > 0.0 {
        for v in &mut noise {
            *v /= sum;
        }
    }

    let children: Vec<NodeIdx> = arena.get(root).children.clone();
    for (&child_idx, &eta) in children.iter().zip(noise.iter()) {
        let child = arena.get_mut(child_idx);
        child.prior = (1.0 - epsilon) * child.prior + epsilon * eta;
    }
}

// ---------------------------------------------------------------------------
// Virtual loss
// ---------------------------------------------------------------------------

/// Magnitude of virtual loss applied per node on the selection path.
pub const VIRTUAL_LOSS: f32 = 1.0;

/// Walk from `leaf` up to the root via parent links and return the indices
/// (leaf-to-root order).  Used to apply / remove virtual loss on the path.
fn path_to_root(arena: &Arena, leaf: NodeIdx) -> Vec<NodeIdx> {
    let mut path = Vec::new();
    let mut idx = leaf;
    loop {
        path.push(idx);
        let parent = arena.get(idx).parent;
        if parent == super::NO_PARENT {
            break;
        }
        idx = parent;
    }
    path
}

/// Mark every node on `path` as "in flight":
///   visit_count += 1,  total_value += VIRTUAL_LOSS.
///
/// **Sign convention (negamax):** each node stores value from the perspective
/// of the side to move *at that node*, and the PUCT formula at the parent
/// negates the child's mean value: `Q = −child.mean_value()`.
/// To make `Q` smaller (depress the path), we must make `child.mean_value()`
/// *larger*, i.e. add to `total_value` — the opposite of what you'd do in a
/// single-perspective tree.
pub fn apply_virtual_loss(arena: &mut Arena, path: &[NodeIdx]) {
    for &idx in path {
        let n = arena.get_mut(idx);
        n.visit_count += 1;
        n.total_value += VIRTUAL_LOSS;
    }
}

/// Undo the virtual loss applied by `apply_virtual_loss`.
/// Must be called before real backpropagation to avoid double-counting.
pub fn remove_virtual_loss(arena: &mut Arena, path: &[NodeIdx]) {
    for &idx in path {
        let n = arena.get_mut(idx);
        n.visit_count = n.visit_count.saturating_sub(1);
        n.total_value -= VIRTUAL_LOSS;
    }
}

// ---------------------------------------------------------------------------
// Temperature-based move selection
// ---------------------------------------------------------------------------

/// Choose a move from `root`'s children using the visit-count distribution
/// raised to the power `1/τ`.
///
/// τ = 0  → greedy: always pick the child with the highest visit count.
/// τ > 0  → sample: move `a` is chosen with probability ∝ N(a)^(1/τ).
///
/// Returns `None` if the root has no children.
pub fn select_move_by_temperature<R: Rng>(
    arena: &Arena,
    root: NodeIdx,
    temperature: f32,
    rng: &mut R,
) -> Option<Move> {
    let children = &arena.get(root).children;
    if children.is_empty() {
        return None;
    }

    if temperature < 1e-6 {
        // Greedy: argmax visit count.
        return children
            .iter()
            .copied()
            .max_by_key(|&idx| arena.get(idx).visit_count)
            .and_then(|idx| arena.get(idx).mv);
    }

    let inv_temp = 1.0 / temperature;
    let weights: Vec<f32> = children
        .iter()
        .map(|&idx| (arena.get(idx).visit_count as f32).powf(inv_temp))
        .collect();

    // WeightedIndex::new returns Err if all weights are zero (no visits yet).
    // Fall back to uniform in that case.
    let chosen = match WeightedIndex::new(&weights) {
        Ok(dist) => children[dist.sample(rng)],
        Err(_) => children[rng.gen_range(0..children.len())],
    };
    arena.get(chosen).mv
}

// ---------------------------------------------------------------------------
// Simulation loop
// ---------------------------------------------------------------------------

/// Run `num_simulations` MCTS iterations from `board` and return the best move.
///
/// The root is expanded once before the loop so that Dirichlet noise (when
/// enabled) can be applied to all root children before any simulation begins.
///
/// Each iteration:
///   1. **Select** — descend via PUCT to a leaf, applying moves to `board`.
///   2. **Expand** — add children for all legal moves (skipped if terminal).
///   3. **Evaluate** — random rollout to terminal.
///   4. **Backprop** — propagate the result up to the root, flipping sign.
///   5. **Undo** — restore `board` to the root position.
///
/// Returns the child of the root with the highest visit count, which is the
/// move recommended by the search.  Returns `None` only if the root has no
/// legal moves (checkmate at the root).
pub fn mcts_search<R: Rng>(
    arena: &mut Arena,
    board: &mut Board,
    num_simulations: u32,
    config: &MctsConfig,
    rng: &mut R,
) -> Option<Move> {
    arena.clear();
    let root = arena.alloc(Node::new(None, 1.0, NO_PARENT));

    // Pre-expand root so noise can be applied before any simulation.
    if !expand(arena, root, board) {
        return None; // terminal at root
    }

    if config.dirichlet_noise {
        add_dirichlet_noise(arena, root, config.dirichlet_alpha, config.dirichlet_epsilon, rng);
    }

    for _ in 0..num_simulations {
        // 1. Selection
        let (leaf, undo_stack) = select(arena, root, board, config.c_puct);

        // 2. Virtual loss — marks path as in-flight for parallel workers.
        let path = path_to_root(arena, leaf);
        apply_virtual_loss(arena, &path);

        // 3. Expansion
        let expanded = expand(arena, leaf, board);

        // 4. Evaluation
        let value = if expanded {
            rollout(board, rng, config.rollout_depth)
        } else {
            -1.0 // terminal: side to move has no moves → they lose
        };

        // 5. Remove virtual loss, then backpropagate real result.
        remove_virtual_loss(arena, &path);
        backprop(arena, leaf, value);

        // 6. Undo moves to restore board to root position.
        for (mv, undo) in undo_stack.into_iter().rev() {
            unmake_move_full(board, mv, &undo);
        }
    }

    select_move_by_temperature(arena, root, config.temperature, rng)
}

// ---------------------------------------------------------------------------
// Parallel simulation loop
// ---------------------------------------------------------------------------

/// Run `num_simulations` MCTS iterations across `num_threads` rayon workers.
///
/// **Locking protocol** — three short critical sections per simulation; the
/// expensive rollout runs lock-free between them:
///
/// 1. **Select + apply VL** (locked): descend tree, mark path in-flight.
/// 2. **Expand** (locked): add child nodes for the leaf position.
/// 3. **Remove VL + backprop** (locked): correct the virtual loss, record result.
///
/// Virtual loss between steps 1 and 3 ensures that concurrent threads see
/// the in-flight path as unattractive and choose different branches.
///
/// Returns `None` only if the root has no legal moves.
pub fn mcts_search_parallel(
    board: &Board,
    num_simulations: u32,
    config: &MctsConfig,
    num_threads: usize,
) -> Option<Move> {
    let arena: Arc<Mutex<Arena>> = Arc::new(Mutex::new(Arena::new(500_000)));
    const ROOT: NodeIdx = 0;

    // Single-threaded setup: root allocation, pre-expansion, optional noise.
    {
        let mut a = arena.lock().unwrap();
        a.alloc(Node::new(None, 1.0, NO_PARENT));
        if !expand(&mut a, ROOT, &mut board.clone()) {
            return None;
        }
        if config.dirichlet_noise {
            let mut rng = rand::thread_rng();
            add_dirichlet_noise(&mut a, ROOT, config.dirichlet_alpha, config.dirichlet_epsilon, &mut rng);
        }
    }

    let sims_per_thread = (num_simulations as usize).div_ceil(num_threads.max(1));

    rayon::scope(|s| {
        for _ in 0..num_threads {
            let arena = Arc::clone(&arena);
            let root_board = board.clone();
            let config = config.clone();
            s.spawn(move |_| {
                let mut rng = rand::thread_rng();

                for _ in 0..sims_per_thread {
                    // --- Phase 1 (locked): select, path, apply VL ---
                    let (leaf, move_list, vl_path) = {
                        let mut a = arena.lock().unwrap();
                        let mut tmp = root_board.clone();
                        let (leaf, undo) = select(&a, ROOT, &mut tmp, config.c_puct);
                        let path = path_to_root(&a, leaf);
                        apply_virtual_loss(&mut a, &path);
                        let moves: Vec<Move> = undo.into_iter().map(|(mv, _)| mv).collect();
                        (leaf, moves, path)
                    };

                    // Replay moves onto a fresh board clone to reach the leaf
                    // position — no lock needed, this is purely local work.
                    let mut leaf_board = root_board.clone();
                    for &mv in &move_list {
                        make_move_full(&mut leaf_board, mv);
                    }

                    // --- Phase 2 (locked): expand leaf ---
                    let expanded = {
                        let mut a = arena.lock().unwrap();
                        expand(&mut a, leaf, &mut leaf_board)
                    };

                    // Rollout runs completely lock-free.
                    let value = if expanded {
                        rollout(&leaf_board, &mut rng, config.rollout_depth)
                    } else {
                        -1.0
                    };

                    // --- Phase 3 (locked): remove VL, backprop ---
                    {
                        let mut a = arena.lock().unwrap();
                        remove_virtual_loss(&mut a, &vl_path);
                        backprop(&mut a, leaf, value);
                    }
                }
            });
        }
    });

    let a = arena.lock().unwrap();
    select_move_by_temperature(&a, ROOT, config.temperature, &mut rand::thread_rng())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::mcts::{MctsConfig, Node, NO_PARENT};
    use crate::movegen::generate_legal_moves;
    use crate::moves::unmake_move_full;

    /// Build a root node in a fresh arena, return (arena, root_idx).
    fn make_root(board: &Board) -> (Arena, NodeIdx) {
        let _ = board; // board not needed for root alloc, kept for clarity
        let mut arena = Arena::new(64);
        let root = arena.alloc(Node::new(None, 1.0, NO_PARENT));
        (arena, root)
    }

    #[test]
    fn test_select_root_is_leaf_returns_root() {
        let board = Board::startpos();
        let (arena, root) = make_root(&board);
        let mut b = board.clone();
        let (leaf, undo_stack) = select(&arena, root, &mut b, 1.0);
        assert_eq!(leaf, root);
        assert!(undo_stack.is_empty());
    }

    #[test]
    fn test_select_board_unchanged_when_root_is_leaf() {
        let board = Board::startpos();
        let (arena, root) = make_root(&board);
        let mut b = board.clone();
        select(&arena, root, &mut b, 1.0);
        assert_eq!(b.hash, board.hash);
    }

    #[test]
    fn test_select_descends_to_only_child() {
        // Expand the root with one real legal move, then select.
        let board = Board::startpos();
        let (mut arena, root) = make_root(&board);

        let mut moves = Vec::new();
        generate_legal_moves(&mut board.clone(), &mut moves);
        let mv = moves[0];

        let child = arena.alloc(Node::new(Some(mv), 1.0, root));
        arena.get_mut(root).children.push(child);
        // Give the root a visit count so sqrt(N) > 0
        arena.get_mut(root).visit_count = 1;

        let mut b = board.clone();
        let (leaf, undo_stack) = select(&arena, root, &mut b, 1.0);

        assert_eq!(leaf, child, "should descend to the single child");
        assert_eq!(undo_stack.len(), 1, "one move applied");
        assert_ne!(b.hash, board.hash, "board must reflect the applied move");
    }

    #[test]
    fn test_select_undo_stack_restores_board() {
        let board = Board::startpos();
        let (mut arena, root) = make_root(&board);

        let mut moves = Vec::new();
        generate_legal_moves(&mut board.clone(), &mut moves);
        let mv = moves[0];

        let child = arena.alloc(Node::new(Some(mv), 1.0, root));
        arena.get_mut(root).children.push(child);
        arena.get_mut(root).visit_count = 1;

        let mut b = board.clone();
        let (_leaf, undo_stack) = select(&arena, root, &mut b, 1.0);

        // Reverse the undo stack to restore the board.
        for (mv, undo) in undo_stack.into_iter().rev() {
            unmake_move_full(&mut b, mv, &undo);
        }
        assert_eq!(b.hash, board.hash, "board must be fully restored after undoing");
    }

    #[test]
    fn test_select_two_levels_deep() {
        // root → child_a → grandchild; selection should reach grandchild.
        let board = Board::startpos();
        let (mut arena, root) = make_root(&board);

        let mut moves = Vec::new();
        generate_legal_moves(&mut board.clone(), &mut moves);
        let mv1 = moves[0];

        // Apply first move to get second position's moves.
        let mut b = board.clone();
        let undo1 = make_move_full(&mut b, mv1);
        let mut moves2 = Vec::new();
        generate_legal_moves(&mut b, &mut moves2);
        let mv2 = moves2[0];
        unmake_move_full(&mut b, mv1, &undo1);

        let child = arena.alloc(Node::new(Some(mv1), 1.0, root));
        let grandchild = arena.alloc(Node::new(Some(mv2), 1.0, child));
        arena.get_mut(root).children.push(child);
        arena.get_mut(root).visit_count = 1;
        arena.get_mut(child).children.push(grandchild);
        arena.get_mut(child).visit_count = 1;

        let (leaf, undo_stack) = select(&arena, root, &mut b, 1.0);

        assert_eq!(leaf, grandchild);
        assert_eq!(undo_stack.len(), 2);

        for (mv, undo) in undo_stack.into_iter().rev() {
            unmake_move_full(&mut b, mv, &undo);
        }
        assert_eq!(b.hash, board.hash);
    }

    // --- Expansion tests ---

    #[test]
    fn test_expand_startpos_creates_30_children() {
        let mut board = Board::startpos();
        let mut arena = Arena::new(64);
        let root = arena.alloc(Node::new(None, 1.0, NO_PARENT));

        let expanded = expand(&mut arena, root, &mut board);

        assert!(expanded, "startpos is not terminal");
        assert_eq!(arena.get(root).children.len(), 30);
    }

    #[test]
    fn test_expand_uniform_prior() {
        let mut board = Board::startpos();
        let mut arena = Arena::new(64);
        let root = arena.alloc(Node::new(None, 1.0, NO_PARENT));
        expand(&mut arena, root, &mut board);

        let expected = 1.0 / 30.0_f32;
        for &child_idx in &arena.get(root).children.clone() {
            let p = arena.get(child_idx).prior;
            assert!((p - expected).abs() < 1e-6, "prior {p} != {expected}");
        }
    }

    #[test]
    fn test_expand_children_have_leaf_as_parent() {
        let mut board = Board::startpos();
        let mut arena = Arena::new(64);
        let root = arena.alloc(Node::new(None, 1.0, NO_PARENT));
        expand(&mut arena, root, &mut board);

        for &child_idx in &arena.get(root).children.clone() {
            assert_eq!(arena.get(child_idx).parent, root);
        }
    }

    #[test]
    fn test_expand_leaf_becomes_interior_node() {
        let mut board = Board::startpos();
        let mut arena = Arena::new(64);
        let root = arena.alloc(Node::new(None, 1.0, NO_PARENT));
        assert!(arena.get(root).is_leaf());
        expand(&mut arena, root, &mut board);
        assert!(!arena.get(root).is_leaf());
    }

    #[test]
    fn test_expand_children_carry_moves() {
        let mut board = Board::startpos();
        let mut arena = Arena::new(64);
        let root = arena.alloc(Node::new(None, 1.0, NO_PARENT));
        expand(&mut arena, root, &mut board);

        for &child_idx in &arena.get(root).children.clone() {
            assert!(arena.get(child_idx).mv.is_some(), "every child must have a move");
        }
    }

    #[test]
    fn test_expand_board_unchanged_after_expand() {
        let mut board = Board::startpos();
        let hash_before = board.hash;
        let mut arena = Arena::new(64);
        let root = arena.alloc(Node::new(None, 1.0, NO_PARENT));
        expand(&mut arena, root, &mut board);
        assert_eq!(board.hash, hash_before, "expand must not modify the board");
    }

    // --- Rollout tests ---

    fn seeded_rng(seed: u64) -> rand::rngs::StdRng {
        use rand::SeedableRng;
        rand::rngs::StdRng::seed_from_u64(seed)
    }

    #[test]
    fn test_rollout_does_not_modify_board() {
        let board = Board::startpos();
        let hash_before = board.hash;
        let mut rng = seeded_rng(0);
        rollout(&board, &mut rng, DEFAULT_ROLLOUT_DEPTH);
        assert_eq!(board.hash, hash_before);
    }

    #[test]
    fn test_rollout_value_in_valid_range() {
        let board = Board::startpos();
        let mut rng = seeded_rng(1);
        let v = rollout(&board, &mut rng, DEFAULT_ROLLOUT_DEPTH);
        assert!(v == -1.0 || v == 0.0 || v == 1.0,
            "rollout value must be -1, 0, or 1; got {v}");
    }

    #[test]
    fn test_rollout_deterministic_with_same_seed() {
        let board = Board::startpos();
        let v1 = rollout(&board, &mut seeded_rng(99), DEFAULT_ROLLOUT_DEPTH);
        let v2 = rollout(&board, &mut seeded_rng(99), DEFAULT_ROLLOUT_DEPTH);
        assert_eq!(v1, v2, "same seed must produce the same outcome");
    }

    #[test]
    fn test_rollout_different_seeds_may_differ() {
        // With 30 possible first moves and deep random play, two different seeds
        // will almost certainly produce different results over many samples.
        let board = Board::startpos();
        let outcomes: Vec<f32> = (0..20)
            .map(|s| rollout(&board, &mut seeded_rng(s), DEFAULT_ROLLOUT_DEPTH))
            .collect();
        let all_same = outcomes.windows(2).all(|w| w[0] == w[1]);
        assert!(!all_same, "different seeds should not all produce identical outcomes");
    }

    #[test]
    fn test_rollout_max_depth_zero_returns_draw() {
        let board = Board::startpos();
        let mut rng = seeded_rng(0);
        let v = rollout(&board, &mut rng, 0);
        assert_eq!(v, 0.0, "depth=0 should immediately return draw");
    }

    // --- Backprop tests ---

    fn three_node_arena() -> (Arena, NodeIdx, NodeIdx, NodeIdx) {
        // root (Black) → child (White) → grandchild (Black)
        let mut arena = Arena::new(8);
        let root  = arena.alloc(Node::new(None, 1.0, NO_PARENT));
        let child = arena.alloc(Node::new(None, 1.0, root));
        let gc    = arena.alloc(Node::new(None, 1.0, child));
        arena.get_mut(root).children.push(child);
        arena.get_mut(child).children.push(gc);
        (arena, root, child, gc)
    }

    #[test]
    fn test_backprop_increments_visit_counts() {
        let (mut arena, root, child, gc) = three_node_arena();
        backprop(&mut arena, gc, 1.0);
        assert_eq!(arena.get(gc).visit_count,    1);
        assert_eq!(arena.get(child).visit_count, 1);
        assert_eq!(arena.get(root).visit_count,  1);
    }

    #[test]
    fn test_backprop_sign_alternates() {
        // Value at leaf is +1.0 (good for the leaf's side = Black).
        // child (White's turn) should see −1.0.
        // root  (Black's turn) should see +1.0 again.
        let (mut arena, root, child, gc) = three_node_arena();
        backprop(&mut arena, gc, 1.0);
        assert!((arena.get(gc).total_value    -  1.0).abs() < 1e-6);
        assert!((arena.get(child).total_value - -1.0).abs() < 1e-6);
        assert!((arena.get(root).total_value  -  1.0).abs() < 1e-6);
    }

    #[test]
    fn test_backprop_from_root_only_updates_root() {
        let (mut arena, root, child, _gc) = three_node_arena();
        backprop(&mut arena, root, 0.5);
        assert_eq!(arena.get(root).visit_count,  1);
        assert_eq!(arena.get(child).visit_count, 0, "child must not be touched");
    }

    #[test]
    fn test_backprop_accumulates_across_multiple_calls() {
        let (mut arena, root, _child, gc) = three_node_arena();
        backprop(&mut arena, gc,  1.0);
        backprop(&mut arena, gc, -1.0); // draw/loss from leaf's perspective
        assert_eq!(arena.get(gc).visit_count,   2);
        assert_eq!(arena.get(root).visit_count, 2);
        // net total_value at root: +1.0 + (−1.0) = 0.0
        assert!((arena.get(root).total_value).abs() < 1e-6);
    }

    #[test]
    fn test_backprop_mean_value_correct_after_backprop() {
        // Two wins for the leaf side → Q at leaf = 1.0, Q at child = −1.0.
        let (mut arena, _root, child, gc) = three_node_arena();
        backprop(&mut arena, gc, 1.0);
        backprop(&mut arena, gc, 1.0);
        assert!((arena.get(gc).mean_value()    -  1.0).abs() < 1e-6);
        assert!((arena.get(child).mean_value() - -1.0).abs() < 1e-6);
    }

    // --- Simulation loop tests ---

    #[test]
    fn test_mcts_returns_legal_move() {
        let mut board = Board::startpos();
        let mut arena = Arena::new(4096);
        let mut rng = seeded_rng(0);

        let mv = mcts_search(&mut arena, &mut board, 50, &MctsConfig::default(), &mut rng);
        assert!(mv.is_some(), "must return a move from startpos");

        // Verify the move is actually legal.
        let mut legal = Vec::new();
        generate_legal_moves(&mut board.clone(), &mut legal);
        assert!(legal.contains(&mv.unwrap()), "returned move must be legal");
    }

    #[test]
    fn test_mcts_board_unchanged_after_search() {
        let mut board = Board::startpos();
        let hash_before = board.hash;
        let mut arena = Arena::new(4096);
        let mut rng = seeded_rng(1);

        mcts_search(&mut arena, &mut board, 50, &MctsConfig::default(), &mut rng);
        assert_eq!(board.hash, hash_before, "board must be restored after search");
    }

    #[test]
    fn test_mcts_root_visit_count_equals_simulations() {
        let mut board = Board::startpos();
        let mut arena = Arena::new(4096);
        let mut rng = seeded_rng(2);

        mcts_search(&mut arena, &mut board, 100, &MctsConfig::default(), &mut rng);
        assert_eq!(arena.get(arena.root()).visit_count, 100);
    }

    // --- Dirichlet noise tests ---

    #[test]
    fn test_dirichlet_noise_changes_priors() {
        let mut board = Board::startpos();
        let mut arena = Arena::new(64);
        let root = arena.alloc(Node::new(None, 1.0, NO_PARENT));
        expand(&mut arena, root, &mut board);

        let prior_before: Vec<f32> = arena
            .get(root)
            .children
            .iter()
            .map(|&i| arena.get(i).prior)
            .collect();

        let mut rng = seeded_rng(7);
        add_dirichlet_noise(&mut arena, root, 0.15, 0.25, &mut rng);

        let prior_after: Vec<f32> = arena
            .get(root)
            .children
            .iter()
            .map(|&i| arena.get(i).prior)
            .collect();

        // At least one prior must have changed.
        let any_changed = prior_before
            .iter()
            .zip(prior_after.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(any_changed, "Dirichlet noise must alter at least one prior");
    }

    #[test]
    fn test_dirichlet_noise_priors_sum_to_one() {
        let mut board = Board::startpos();
        let mut arena = Arena::new(64);
        let root = arena.alloc(Node::new(None, 1.0, NO_PARENT));
        expand(&mut arena, root, &mut board);

        let mut rng = seeded_rng(8);
        add_dirichlet_noise(&mut arena, root, 0.15, 0.25, &mut rng);

        let sum: f32 = arena
            .get(root)
            .children
            .iter()
            .map(|&i| arena.get(i).prior)
            .sum();
        assert!((sum - 1.0).abs() < 1e-4, "priors must still sum to ~1 after noise; got {sum}");
    }

    #[test]
    fn test_dirichlet_noise_priors_all_positive() {
        let mut board = Board::startpos();
        let mut arena = Arena::new(64);
        let root = arena.alloc(Node::new(None, 1.0, NO_PARENT));
        expand(&mut arena, root, &mut board);

        let mut rng = seeded_rng(9);
        add_dirichlet_noise(&mut arena, root, 0.15, 0.25, &mut rng);

        for &child_idx in &arena.get(root).children.clone() {
            let p = arena.get(child_idx).prior;
            assert!(p >= 0.0, "priors must stay non-negative after noise; got {p}");
        }
    }

    #[test]
    fn test_dirichlet_noise_no_effect_on_empty_root() {
        // Root with no children: noise should be a no-op.
        let mut arena = Arena::new(8);
        let root = arena.alloc(Node::new(None, 1.0, NO_PARENT));
        let mut rng = seeded_rng(10);
        // Must not panic.
        add_dirichlet_noise(&mut arena, root, 0.15, 0.25, &mut rng);
        assert!(arena.get(root).children.is_empty());
    }

    #[test]
    fn test_mcts_noise_disabled_by_default() {
        // With dirichlet_noise = false (default), priors at root children
        // must remain uniform (1/N) after search.
        let mut board = Board::startpos();
        let mut arena = Arena::new(4096);
        let mut rng = seeded_rng(11);
        let config = MctsConfig::default();
        assert!(!config.dirichlet_noise);

        mcts_search(&mut arena, &mut board, 10, &config, &mut rng);

        let expected = 1.0 / 30.0_f32;
        for &child_idx in &arena.get(arena.root()).children.clone() {
            let p = arena.get(child_idx).prior;
            assert!((p - expected).abs() < 1e-6, "prior should be uniform; got {p}");
        }
    }

    // --- Virtual loss tests ---

    #[test]
    fn test_apply_vl_increments_visit_and_value() {
        // In negamax, VL adds to total_value (not subtracts) so that the
        // parent's PUCT Q = -mean_value() goes negative, deterring selection.
        let mut arena = Arena::new(8);
        let root  = arena.alloc(Node::new(None, 1.0, NO_PARENT));
        let child = arena.alloc(Node::new(None, 0.5, root));
        arena.get_mut(root).children.push(child);

        apply_virtual_loss(&mut arena, &[root, child]);

        assert_eq!(arena.get(root).visit_count,  1);
        assert_eq!(arena.get(child).visit_count, 1);
        assert!((arena.get(root).total_value  - VIRTUAL_LOSS).abs() < 1e-6);
        assert!((arena.get(child).total_value - VIRTUAL_LOSS).abs() < 1e-6);
    }

    #[test]
    fn test_remove_vl_restores_zero_state() {
        let mut arena = Arena::new(8);
        let root = arena.alloc(Node::new(None, 1.0, NO_PARENT));
        apply_virtual_loss(&mut arena, &[root]);
        remove_virtual_loss(&mut arena, &[root]);
        assert_eq!(arena.get(root).visit_count, 0);
        assert!(arena.get(root).total_value.abs() < 1e-6);
    }

    #[test]
    fn test_vl_apply_remove_then_backprop_equals_plain_backprop() {
        // apply VL → remove VL → backprop  should equal  plain backprop.
        let mut arena1 = Arena::new(4);
        let r1 = arena1.alloc(Node::new(None, 1.0, NO_PARENT));
        apply_virtual_loss(&mut arena1, &[r1]);
        remove_virtual_loss(&mut arena1, &[r1]);
        backprop(&mut arena1, r1, 0.7);

        let mut arena2 = Arena::new(4);
        let r2 = arena2.alloc(Node::new(None, 1.0, NO_PARENT));
        backprop(&mut arena2, r2, 0.7);

        assert_eq!(arena1.get(r1).visit_count, arena2.get(r2).visit_count);
        assert!((arena1.get(r1).total_value - arena2.get(r2).total_value).abs() < 1e-6);
    }

    #[test]
    fn test_path_to_root_single_node() {
        let mut arena = Arena::new(4);
        let root = arena.alloc(Node::new(None, 1.0, NO_PARENT));
        let path = path_to_root(&arena, root);
        assert_eq!(path, vec![root]);
    }

    #[test]
    fn test_path_to_root_depth_two() {
        let mut arena = Arena::new(8);
        let root  = arena.alloc(Node::new(None, 1.0, NO_PARENT));
        let child = arena.alloc(Node::new(None, 0.5, root));
        arena.get_mut(root).children.push(child);
        let path = path_to_root(&arena, child);
        assert_eq!(path, vec![child, root]);
    }

    #[test]
    fn test_vl_steers_second_selection_away() {
        // After thread A applies VL to child 0, thread B (running select on
        // the same tree) should prefer child 1 (no VL penalty).
        let mut board = Board::startpos();
        let mut arena = Arena::new(256);
        let root = arena.alloc(Node::new(None, 1.0, NO_PARENT));
        expand(&mut arena, root, &mut board);
        arena.get_mut(root).visit_count = 10; // non-zero so PUCT U term is live

        // Apply VL to the first child simulating thread A in-flight.
        let first_child = arena.get(root).children[0];
        apply_virtual_loss(&mut arena, &[first_child]);

        // Thread B selects: it should NOT pick first_child.
        let mut b = board.clone();
        let (leaf, _) = select(&arena, root, &mut b, 1.0);
        assert_ne!(leaf, first_child, "VL should steer selection away from in-flight child");
    }

    // --- Parallel search tests ---

    #[test]
    fn test_mcts_parallel_returns_legal_move() {
        let board = Board::startpos();
        let mv = mcts_search_parallel(&board, 100, &MctsConfig::default(), 2);
        assert!(mv.is_some());
        let mut legal = Vec::new();
        generate_legal_moves(&mut board.clone(), &mut legal);
        assert!(legal.contains(&mv.unwrap()));
    }

    #[test]
    fn test_mcts_parallel_board_unchanged() {
        let board = Board::startpos();
        let hash_before = board.hash;
        mcts_search_parallel(&board, 100, &MctsConfig::default(), 2);
        assert_eq!(board.hash, hash_before);
    }

    #[test]
    fn test_mcts_parallel_single_thread_returns_legal_move() {
        let board = Board::startpos();
        let mv = mcts_search_parallel(&board, 50, &MctsConfig::default(), 1);
        assert!(mv.is_some());
        let mut legal = Vec::new();
        generate_legal_moves(&mut board.clone(), &mut legal);
        assert!(legal.contains(&mv.unwrap()));
    }

    #[test]
    fn test_mcts_parallel_four_threads_returns_legal_move() {
        let board = Board::startpos();
        let mv = mcts_search_parallel(&board, 200, &MctsConfig::default(), 4);
        assert!(mv.is_some());
        let mut legal = Vec::new();
        generate_legal_moves(&mut board.clone(), &mut legal);
        assert!(legal.contains(&mv.unwrap()));
    }

    // --- Temperature selection tests ---

    /// Build a root with children whose visit counts are set manually.
    fn arena_with_visit_counts(counts: &[u32]) -> (Arena, NodeIdx) {
        let mut arena = Arena::new(64);
        let root = arena.alloc(Node::new(None, 1.0, NO_PARENT));
        for (i, &n) in counts.iter().enumerate() {
            let child = arena.alloc(Node::new(
                // Use a dummy move value — only identity matters for these tests.
                Some(crate::types::Move(i as u32 + 1)),
                1.0 / counts.len() as f32,
                root,
            ));
            arena.get_mut(child).visit_count = n;
            arena.get_mut(root).children.push(child);
        }
        (arena, root)
    }

    #[test]
    fn test_temperature_zero_picks_most_visited() {
        // counts: child 0 = 5, child 1 = 20, child 2 = 1.  Child 1 must win.
        let (arena, root) = arena_with_visit_counts(&[5, 20, 1]);
        let mut rng = seeded_rng(20);
        let mv = select_move_by_temperature(&arena, root, 0.0, &mut rng);
        // child 1 is arena index 2 (root=0, child0=1, child1=2).
        assert_eq!(mv, arena.get(2).mv, "greedy must pick child with visit_count=20");
    }

    #[test]
    fn test_temperature_zero_deterministic() {
        let (arena, root) = arena_with_visit_counts(&[3, 10, 1]);
        let mv1 = select_move_by_temperature(&arena, root, 0.0, &mut seeded_rng(0));
        let mv2 = select_move_by_temperature(&arena, root, 0.0, &mut seeded_rng(99));
        assert_eq!(mv1, mv2, "greedy must be deterministic regardless of seed");
    }

    #[test]
    fn test_temperature_one_samples_proportionally() {
        // Counts heavily skewed: child 0 = 1, child 1 = 999.
        // With τ=1 and many samples, child 1 should be picked ~99.9% of the time.
        let (arena, root) = arena_with_visit_counts(&[1, 999]);
        let mut rng = seeded_rng(42);
        let child_1_mv = arena.get(2).mv; // root=0, child0=1, child1=2

        let n = 200u32;
        let child_1_count = (0..n)
            .filter(|_| select_move_by_temperature(&arena, root, 1.0, &mut rng) == child_1_mv)
            .count();

        assert!(
            child_1_count > 180,
            "child with 999 visits should be chosen >90% of the time; got {child_1_count}/{n}"
        );
    }

    #[test]
    fn test_temperature_high_flattens_distribution() {
        // Counts: child 0 = 1, child 1 = 1000.
        // At τ=10 the distribution should be much flatter than τ=1.
        let (arena, root) = arena_with_visit_counts(&[1, 1000]);
        let mut rng = seeded_rng(55);
        let child_1_mv = arena.get(2).mv;

        let n = 200u32;
        let count_high_t = (0..n)
            .filter(|_| select_move_by_temperature(&arena, root, 10.0, &mut rng) == child_1_mv)
            .count();

        // With τ=1 we'd expect ~99.9% child 1; with τ=10 it should be noticeably less.
        assert!(
            count_high_t < 200,
            "high temperature should occasionally pick the less-visited child"
        );
    }

    #[test]
    fn test_temperature_none_for_empty_root() {
        let mut arena = Arena::new(8);
        let root = arena.alloc(Node::new(None, 1.0, NO_PARENT));
        let mut rng = seeded_rng(0);
        let mv = select_move_by_temperature(&arena, root, 1.0, &mut rng);
        assert!(mv.is_none());
    }

    #[test]
    fn test_mcts_default_temperature_zero_returns_best() {
        // With τ=0 the returned move must be the most-visited root child.
        let mut board = Board::startpos();
        let mut arena = Arena::new(4096);
        let mut rng = seeded_rng(30);

        let mv = mcts_search(&mut arena, &mut board, 100, &MctsConfig::default(), &mut rng);

        let expected = arena
            .get(arena.root())
            .children
            .iter()
            .copied()
            .max_by_key(|&i| arena.get(i).visit_count)
            .and_then(|i| arena.get(i).mv);

        assert_eq!(mv, expected, "τ=0 must return the most-visited move");
    }

    #[test]
    fn test_mcts_temperature_one_still_legal() {
        let mut board = Board::startpos();
        let mut arena = Arena::new(4096);
        let mut rng = seeded_rng(31);
        let config = MctsConfig { temperature: 1.0, ..MctsConfig::default() };

        let mv = mcts_search(&mut arena, &mut board, 100, &config, &mut rng);
        assert!(mv.is_some());

        let mut legal = Vec::new();
        generate_legal_moves(&mut board.clone(), &mut legal);
        assert!(legal.contains(&mv.unwrap()), "temperature=1 must still return a legal move");
    }

    #[test]
    fn test_mcts_noise_enabled_changes_root_priors() {
        let mut board = Board::startpos();
        let mut arena = Arena::new(4096);
        let mut rng = seeded_rng(12);
        let config = MctsConfig {
            dirichlet_noise: true,
            ..MctsConfig::default()
        };

        mcts_search(&mut arena, &mut board, 10, &config, &mut rng);

        let uniform = 1.0 / 30.0_f32;
        let any_changed = arena
            .get(arena.root())
            .children
            .iter()
            .any(|&i| (arena.get(i).prior - uniform).abs() > 1e-6);
        assert!(any_changed, "noise=true must change at least one root prior");
    }

    /// Quick sanity check — not part of CI.
    /// Run with: cargo test mcts_sanity -- --ignored --nocapture
    #[test]
    #[ignore]
    fn mcts_sanity_startpos() {
        use std::time::Instant;

        let num_sims = 500u32;
        let mut board  = Board::startpos();
        let mut arena  = Arena::new(200_000);
        let mut rng    = seeded_rng(42);

        let t0   = Instant::now();
        let best = mcts_search(
            &mut arena, &mut board,
            num_sims, &MctsConfig::default(), &mut rng,
        );
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let sims_per_s = num_sims as f64 / (elapsed_ms / 1000.0);

        println!("\n--- MCTS startpos ({num_sims} sims) ---");
        println!("time     : {elapsed_ms:.1} ms");
        println!("sims/sec : {sims_per_s:.0}");
        println!("nodes    : {}", arena.len());
        println!("best move: {}", best.map(|m| m.to_usi_string()).unwrap_or("none".into()));
        println!();

        // Sort root children by visit count descending.
        let mut children: Vec<NodeIdx> = arena.get(arena.root()).children.clone();
        children.sort_by(|&a, &b| {
            arena.get(b).visit_count.cmp(&arena.get(a).visit_count)
        });

        println!("{:<10}  {:>6}  {:>7}", "move", "visits", "Q(stm)");
        println!("{}", "-".repeat(28));
        for &idx in children.iter().take(10) {
            let node = arena.get(idx);
            // Negate: child stores value from its own (opponent's) perspective.
            let q_for_stm = -node.mean_value();
            println!(
                "{:<10}  {:>6}  {:>+7.3}",
                node.mv.unwrap().to_usi_string(),
                node.visit_count,
                q_for_stm,
            );
        }
        if children.len() > 10 {
            println!("... ({} total children)", children.len());
        }
    }

    #[test]
    fn test_mcts_more_simulations_visits_more_nodes() {
        let mut board = Board::startpos();
        let mut rng1 = seeded_rng(3);
        let mut rng2 = seeded_rng(3);

        let mut arena1 = Arena::new(4096);
        mcts_search(&mut arena1, &mut board, 20, &MctsConfig::default(), &mut rng1);

        let mut arena2 = Arena::new(4096);
        mcts_search(&mut arena2, &mut board, 200, &MctsConfig::default(), &mut rng2);

        assert!(arena2.len() > arena1.len(),
            "more simulations should produce a larger tree");
    }
}
