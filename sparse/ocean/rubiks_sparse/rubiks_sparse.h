/* Rubiks Sparse: a 3x3x3 Rubik's cube, held yellow up and green in front.
 * 18 actions (6 faces: clockwise, counter-clockwise or a half turn).
 * The reward is 1 for solving the cube and nothing else.
 *
 * Random play never solves a scrambled cube, so on scrambles alone there
 * would be nothing to learn from. Most episodes are instead replays: they
 * start from a cube the policy has already solved, turned a little further
 * at random. If it solves that one too, it becomes a start for later
 * replays. The first such cube is the solved cube itself, so the starts move
 * outward from there, and only as fast as the policy can follow.
 *
 * The rest of the episodes are uniformly random scrambles. Those alone are
 * logged, so the dashboard describes random cubes and not the replays.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "raylib.h"
#include "rlgl.h"
typedef unsigned char obs_t;
#include "pufferenv.h"

#define ACT_SIZES {18}
#define OBS_SIZE (54 * 6)
#define NUM_ATNS 1
#define PUF_STEPS_PER_SEC 4
#define RENDER_FRAMES (60 / PUF_STEPS_PER_SEC)
// How many solved starts the pool and the archive hold (see below), and how
// many replays go into each reading of how the replays are going
#define POOL 65536
#define ARCHIVE (1 << 20)
#define BATCH 20000

struct Log {
    float perf;
    float score;
    float shortest_solve;
    float episode_length;
    float replay_perf;
    float replay_length;
    float n;
};

struct Env {
    Log log;
    Agent agents[1];
    int tag;
    int boundary_reached;
    int num_agents;
    int max_steps;
    float fresh_share;
    float archive_share;
    float archive_keep;
    int nudge;
    // Whether this episode is a replay, and the cube it started from
    int replay;
    unsigned char first[54];
    int tick;
    int moves;
    int turned;
    int key;
    float view[3][3];
    unsigned char stickers[54];
    unsigned int rng;
};
typedef Env Rubiks;

// Sticker j = 9*face + cell holds a color 0-5. Faces are +X -X +Y -Y +Z -Z
// (right, left, up, down, front, back) and a solved face f is all color f.
// A sticker sits 2 units out along its face axis and -1..1 along the other
// two, so rotating that one point moves its position and facing together.
void sticker_pos(int j, int* q) {
    int axis = j / 18;
    q[axis] = (j / 9) % 2 ? -2 : 2;
    q[(axis + 1) % 3] = j % 3 - 1;
    q[(axis + 2) % 3] = (j % 9) / 3 - 1;
}

int sticker_at(int* q) {
    int facing = (abs(q[1]) == 2) + 2 * (abs(q[2]) == 2);
    int cell = q[(facing + 1) % 3] + 1 + 3 * (q[(facing + 2) % 3] + 1);
    return 18 * facing + (q[facing] < 0 ? 9 : 0) + cell;
}

// The sticker on the `axis` side of the piece at p
int sticker_on(int* p, int axis) {
    int q[3] = {p[0], p[1], p[2]};
    q[axis] *= 2;
    return sticker_at(q);
}

// Lookup tables, filled once from the geometry above so that stepping never
// has to work it out again.
// A turn of the layer under a face moves 20 stickers: FROM[move][k] goes to
// TO[move][k]. move = 3*face + (0 clockwise, 1 counter-clockwise, 2 half
// turn).
// CORNERS[i] and EDGES[i] are the stickers of the i-th corner and edge slot.
// A corner's start with the one facing up or down and go the same way round
// every corner. An edge's start with the one facing up or down, or front or
// back for the middle layer's.
unsigned char FROM[18][20];
unsigned char TO[18][20];
unsigned char CORNERS[8][3];
unsigned char EDGES[12][2];

void build_tables() {
    for (int move = 0; move < 18; move++) {
        int axis = move / 6;
        int side = (move / 3) % 2 ? -1 : 1;
        int spin = move % 3 == 1 ? side : -side;
        int quarters = move % 3 == 2 ? 2 : 1;
        int a = (axis + 1) % 3;
        int b = (axis + 2) % 3;
        int moved = 0;
        for (int j = 0; j < 54; j++) {
            int q[3];
            sticker_pos(j, q);
            if (q[axis] * side <= 0) {
                continue;
            }
            for (int i = 0; i < quarters; i++) {
                int qa = q[a];
                q[a] = -spin * q[b];
                q[b] = spin * qa;
            }
            // The face's own center turns on the spot
            if (sticker_at(q) != j) {
                FROM[move][moved] = j;
                TO[move][moved++] = sticker_at(q);
            }
        }
    }

    int corners = 0;
    int edges = 0;
    for (int c = 0; c < 27; c++) {
        int p[3] = {c % 3 - 1, (c / 3) % 3 - 1, c / 9 - 1};
        int size = abs(p[0]) + abs(p[1]) + abs(p[2]);
        if (size < 2) {
            continue;
        }
        unsigned char* slot = size == 3 ? CORNERS[corners] : EDGES[edges];
        int k = 0;
        for (int i = 1; i <= 3; i++) {
            if (p[i % 3] != 0) {
                slot[k++] = sticker_on(p, i % 3);
            }
        }
        // Mirror-image corners list theirs in the opposite sense
        if (p[0] * p[1] * p[2] < 0) {
            unsigned char tmp = slot[1];
            slot[1] = slot[2];
            slot[2] = tmp;
        }
        corners += size == 3;
        edges += size == 2;
    }
}

void turn(unsigned char* s, int move) {
    unsigned char moved[20];
    for (int k = 0; k < 20; k++) {
        moved[k] = s[FROM[move][k]];
    }
    for (int k = 0; k < 20; k++) {
        s[TO[move][k]] = moved[k];
    }
}

int is_solved(unsigned char* s) {
    int right = 0;
    for (int j = 0; j < 54; j++) {
        right += s[j] == j / 9;
    }
    return right == 54;
}

// One-hot color per sticker
void compute_observations(Rubiks* env) {
    obs_t* obs = env->agents[0].observations;
    memset(obs, 0, OBS_SIZE);
    for (int j = 0; j < 54; j++) {
        obs[6 * j + env->stickers[j]] = 1;
    }
}

// Every reachable state is equally likely. Corners (3 stickers) and then
// edges (2) are shuffled between their slots and twisted in place. Turning
// faces only reaches states where both shuffles have the same parity and
// each kind's twists sum to zero, so the last edge swap and the last twist of
// each kind are forced.
void scramble(Rubiks* env) {
    for (int j = 0; j < 54; j++) {
        env->stickers[j] = j / 9;
    }
    int parity = 0;
    for (int size = 3; size >= 2; size--) {
        unsigned char* slots = size == 3 ? CORNERS[0] : EDGES[0];
        int n = size == 3 ? 8 : 12;
        int perm[12];
        for (int i = 0; i < n; i++) {
            perm[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = rand_r(&env->rng) % (i + 1);
            if (size == 2 && i == 1) {
                j = !parity;
            }
            int tmp = perm[i];
            perm[i] = perm[j];
            perm[j] = tmp;
            parity ^= i != j;
        }

        int twists = 0;
        for (int i = 0; i < n; i++) {
            int twist = rand_r(&env->rng) % size;
            if (i == n - 1) {
                twist = (size - twists % size) % size;
            }
            twists += twist;
            for (int k = 0; k < size; k++) {
                int home = slots[size * perm[i] + k];
                env->stickers[slots[size * i + (k + twist) % size]] = home / 9;
            }
        }
    }
}

// What the envs share. They step on several threads, so all of it is read
// and written under SHARED_LOCK.
//
// The pool is the cubes that episodes started from and went on to solve,
// newest over oldest. It starts as the solved cube alone (see puf_init).
// Every solved start goes in, which rewrites the whole pool every second or
// two. That lets it move outward quickly, but a policy that stumbles for a
// few seconds is then left with only the few cubes it can still solve, and
// everything further out is gone.
// So an archive_keep share of solved starts also go into the archive, each
// in place of a cube picked at random once it is full. That gives it a
// memory of minutes. An archive_share of replays start from there.
char SHARED_LOCK;
unsigned char POOL_CUBES[POOL][54];
unsigned char ARCHIVE_CUBES[ARCHIVE][54];
long POOLED;
long ARCHIVED;
// The batch of replays in hand: how many, how many of them were solved, and
// the moves those solves took
int REPLAYS;
int REPLAYS_SOLVED;
int REPLAY_MOVES;
// The last whole batch: the share solved, and the mean moves of its solves.
// That is the nearest thing there is to how far out the pool has got, since
// a cube cannot be solved in fewer moves than it is from solved.
float REPLAY_PERF;
float REPLAY_LENGTH;
// The fewest moves a random scramble has been solved in, or 0 before the
// first such solve
int SHORTEST_SOLVE;

void lock_shared() {
    while (__atomic_test_and_set(&SHARED_LOCK, __ATOMIC_ACQUIRE)) {
    }
}

void unlock_shared() {
    __atomic_clear(&SHARED_LOCK, __ATOMIC_RELEASE);
}

// A fresh_share of episodes are uniformly random scrambles. The rest are
// replays: a cube from the pool or the archive, given up to `nudge` more
// random turns.
void puf_reset(Rubiks* env) {
    env->tick = 0;
    env->turned = -1;
    env->replay = rand_r(&env->rng) >= env->fresh_share * RAND_MAX;
    if (env->replay) {
        int archive = rand_r(&env->rng) < env->archive_share * RAND_MAX;
        long slot = rand_r(&env->rng);
        lock_shared();
        if (archive && ARCHIVED > 0) {
            memcpy(env->stickers, ARCHIVE_CUBES[slot % ARCHIVED], 54);
        } else {
            slot %= POOLED < POOL ? POOLED : POOL;
            memcpy(env->stickers, POOL_CUBES[slot], 54);
        }
        unlock_shared();
        // At least one turn if that leaves it solved
        int turns = rand_r(&env->rng) % (env->nudge + 1);
        for (int i = 0; i < turns || is_solved(env->stickers); i++) {
            turn(env->stickers, rand_r(&env->rng) % 18);
        }
    } else {
        do {
            scramble(env);
        } while (is_solved(env->stickers));
    }
    memcpy(env->first, env->stickers, 54);
    compute_observations(env);
}

// The end of an episode. A start that was solved joins the pool, and perhaps
// the archive, so that later replays begin a little further out again. Starts
// the policy could not solve are left out, which keeps the pool at the edge
// of what it can do. Fresh scrambles that were solved go in as well.
void finish_episode(Rubiks* env, int solved) {
    int keep = 0;
    long slot = 0;
    if (solved) {
        keep = rand_r(&env->rng) < env->archive_keep * RAND_MAX;
        slot = rand_r(&env->rng) % ARCHIVE;
    }
    lock_shared();
    if (solved) {
        memcpy(POOL_CUBES[POOLED++ % POOL], env->first, 54);
        if (keep) {
            if (ARCHIVED < ARCHIVE) {
                slot = ARCHIVED++;
            }
            memcpy(ARCHIVE_CUBES[slot], env->first, 54);
        }
    }
    if (env->replay) {
        // Replays stay out of the log, which describes random scrambles
        REPLAYS++;
        REPLAYS_SOLVED += solved;
        REPLAY_MOVES += solved ? env->tick : 0;
        if (REPLAYS == BATCH) {
            REPLAY_PERF = (float)REPLAYS_SOLVED / BATCH;
            REPLAY_LENGTH = (float)REPLAY_MOVES
                / (REPLAYS_SOLVED > 0 ? REPLAYS_SOLVED : 1);
            REPLAYS = 0;
            REPLAYS_SOLVED = 0;
            REPLAY_MOVES = 0;
        }
    } else {
        if (solved && (SHORTEST_SOLVE == 0 || env->tick < SHORTEST_SOLVE)) {
            SHORTEST_SOLVE = env->tick;
        }
        // perf and score are the same thing here: there is no part credit
        env->log.perf += solved;
        env->log.score += solved;
        env->log.episode_length += env->tick;
        // The next three are logged as they stand by every episode, so that
        // the mean the dashboard takes over episodes comes out as the value
        // itself. Until there is a solve the record reads as the move limit.
        env->log.shortest_solve += SHORTEST_SOLVE ? SHORTEST_SOLVE
            : env->max_steps;
        env->log.replay_perf += REPLAY_PERF;
        env->log.replay_length += REPLAY_LENGTH;
        env->log.n++;
    }
    unlock_shared();
}

void puf_step(Rubiks* env) {
    // puf_render latches a key while Left Shift is held. No time limit then.
    int human = IsWindowReady() && IsKeyDown(KEY_LEFT_SHIFT);
    int move = env->agents[0].actions[0];
    if (human) {
        move = env->key - 1;
        env->key = 0;
    }
    env->agents[0].rewards[0] = 0;
    env->agents[0].terminals[0] = 0;
    env->turned = move;
    if (move < 0) {
        return;
    }

    turn(env->stickers, move);
    env->tick++;
    // 1 for solving the cube and nothing else
    int solved = is_solved(env->stickers);
    env->agents[0].rewards[0] = solved;
    if (!solved && (human || env->tick < env->max_steps)) {
        compute_observations(env);
        return;
    }

    env->agents[0].terminals[0] = 1;
    finish_episode(env, solved);
    // puf_render still shows a solved cube after the reset, so keep its move
    // count and final turn
    env->moves = env->tick;
    puf_reset(env);
    if (solved) {
        env->turned = move;
    }
}

const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 255};
// Orange, red, yellow, white, green, blue
const Color STICKER_COLORS[6] = {
    {255, 110, 0, 255},
    {187, 0, 0, 255},
    {255, 213, 0, 255},
    {241, 241, 241, 255},
    {0, 155, 72, 255},
    {0, 70, 173, 255},
};

void draw_box(float* pos, float* size, Color color) {
    DrawCubeV((Vector3){pos[0], pos[1], pos[2]},
        (Vector3){size[0], size[1], size[2]}, color);
}

// Rotates two perpendicular vectors within the plane they span
void rotate_pair(float* a, float* b, float angle) {
    for (int i = 0; i < 3; i++) {
        float ai = a[i];
        a[i] = cosf(angle) * ai + sinf(angle) * b[i];
        b[i] = cosf(angle) * b[i] - sinf(angle) * ai;
    }
}

// Hold Left Shift and press R L U D F B to turn that face clockwise.
// Hold Space as well for counter-clockwise, or 2 for a half turn. Drag the
// mouse to orbit.
void puf_render(Rubiks* env) {
    if (!IsWindowReady()) {
        InitWindow(720, 720, "PufferLib Rubiks Sparse");
        SetTargetFPS(60);
        // Facing green with yellow up, then moved right and above
        for (int i = 0; i < 3; i++) {
            env->view[i][i] = 1;
        }
        rotate_pair(env->view[0], env->view[2], -0.6f);
        rotate_pair(env->view[1], env->view[2], -0.5f);
    }

    // The solving step already reset the cube. Show it solved and hold there.
    // Only that step ends an episode with a turn left to animate.
    int solved = env->agents[0].terminals[0] > 0 && env->turned >= 0;
    int frames = solved ? 4 * RENDER_FRAMES : RENDER_FRAMES;
    // The last turn is animated by drawing its layer rotated back, then easing
    // that rotation to zero. Same move decoding as build_tables().
    int axis = env->turned < 0 ? 0 : env->turned / 6;
    int side = (env->turned / 3) % 2 ? -1 : 1;
    int spin = env->turned % 3 == 1 ? side : -side;
    int quarters = env->turned % 3 == 2 ? 2 : 1;
    int face_keys[6] = {KEY_R, KEY_L, KEY_U, KEY_D, KEY_F, KEY_B};
    // The camera's own right, up and back axes. It sits out along back.
    float* right = env->view[0];
    float* up = env->view[1];
    float* back = env->view[2];
    for (int f = 0; f < frames; f++) {
        if (IsKeyDown(KEY_ESCAPE)) {
            exit(0);
        }
        for (int face = 0; face < 6; face++) {
            if (IsKeyDown(KEY_LEFT_SHIFT) && IsKeyPressed(face_keys[face])) {
                env->key = 1 + 3 * face
                    + (IsKeyDown(KEY_TWO) ? 2 : IsKeyDown(KEY_SPACE));
            }
        }
        // Orbit about the screen's own axes, so there are no poles to stop at
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            Vector2 drag = GetMouseDelta();
            rotate_pair(right, back, 0.01f * drag.x);
            rotate_pair(up, back, -0.01f * drag.y);
        }
        Camera3D camera = {
            .position = {9 * back[0], 9 * back[1], 9 * back[2]},
            .up = {up[0], up[1], up[2]},
            .fovy = 45,
            .projection = CAMERA_PERSPECTIVE,
        };
        float angle = 0;
        if (env->turned >= 0 && f < RENDER_FRAMES) {
            angle = -90.0f * quarters * spin * (RENDER_FRAMES - 1 - f)
                / RENDER_FRAMES;
        }

        BeginDrawing();
        ClearBackground(PUFF_BACKGROUND);
        BeginMode3D(camera);
        // Three black slabs along the turned axis, each with its stickers
        for (int layer = -1; layer <= 1; layer++) {
            float slab_pos[3] = {0, 0, 0};
            float slab_size[3] = {3, 3, 3};
            slab_pos[axis] = layer;
            slab_size[axis] = 1;
            rlPushMatrix();
            if (layer == side) {
                rlRotatef(angle, axis == 0, axis == 1, axis == 2);
            }
            draw_box(slab_pos, slab_size, BLACK);
            for (int j = 0; j < 54; j++) {
                int q[3];
                sticker_pos(j, q);
                if ((q[axis] > 0) - (q[axis] < 0) != layer) {
                    continue;
                }
                float pos[3] = {q[0], q[1], q[2]};
                float size[3] = {0.9f, 0.9f, 0.9f};
                pos[j / 18] *= 0.75f;
                size[j / 18] = 0.05f;
                draw_box(pos, size,
                    STICKER_COLORS[solved ? j / 9 : env->stickers[j]]);
            }
            rlPopMatrix();
        }
        EndMode3D();
        DrawText("Left Shift + R L U D F B: turn a face",
            10, 10, 20, PUFF_WHITE);
        DrawText("+ Space: reverse, + 2: half turn", 10, 36, 20, PUFF_WHITE);
        DrawText("Drag: orbit", 10, 62, 20, PUFF_WHITE);
        DrawText(TextFormat("Moves: %d", solved ? env->moves : env->tick),
            10, 88, 20, PUFF_WHITE);
        if (solved) {
            DrawText("Solved!", 10, 114, 20, PUFF_WHITE);
        }
        EndDrawing();
        puf_web_vsync();
    }
}

void puf_close(Rubiks* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "shortest_solve", log->shortest_solve);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "replay_perf", log->replay_perf);
    dict_set(out, "replay_length", log->replay_length);
    dict_set(out, "n", log->n);
}

void puf_init(Env* env, Dict* kwargs) {
    build_tables();
    env->num_agents = 1;
    env->max_steps = dict_get(kwargs, "max_steps");
    env->fresh_share = dict_get(kwargs, "fresh_share");
    env->archive_share = dict_get(kwargs, "archive_share");
    env->archive_keep = dict_get(kwargs, "archive_keep");
    env->nudge = dict_get(kwargs, "nudge");
    // The pool starts as the solved cube. Every env comes through here, so it
    // is only put in once.
    lock_shared();
    if (POOLED == 0) {
        for (int j = 0; j < 54; j++) {
            POOL_CUBES[0][j] = j / 9;
        }
        POOLED = 1;
    }
    unlock_shared();
    // The trainer starts each env's generator at its index, so every run
    // would deal the same cubes. seed moves all of them to another stream:
    // 0 leaves them where they are.
    env->rng += 1000003u * (unsigned int)dict_get(kwargs, "seed");
    env->agents[0].action_mask = NULL;
    env->agents[0].policy = 0;
}
