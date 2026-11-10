// Test unaligned atomic memory instruction decoding and emulation (dm/host/crash.c)
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <assert.h>

typedef struct {
    uint64_t regs[31];
    uint64_t sp;
    uint64_t pc;
    uint64_t pstate;
} mock_ucontext_t;

#define PLAT_UC_X(uc, i) ((uc)->regs[i])
#define PLAT_UC_FP(uc) ((uc)->regs[29])
#define PLAT_UC_LR(uc) ((uc)->regs[30])
#define PLAT_UC_SP(uc) ((uc)->sp)
#define PLAT_UC_PC(uc) ((uc)->pc)

static inline uint64_t reg_get_xn(mock_ucontext_t *uc, unsigned r) {
    if (r == 31) return (uint64_t)PLAT_UC_SP(uc);
    if (r == 30) return (uint64_t)PLAT_UC_LR(uc);
    if (r == 29) return (uint64_t)PLAT_UC_FP(uc);
    return (uint64_t)PLAT_UC_X(uc, r);
}

static inline uint64_t reg_get_xt(mock_ucontext_t *uc, unsigned r) {
    if (r == 31) return 0;
    if (r == 30) return (uint64_t)PLAT_UC_LR(uc);
    if (r == 29) return (uint64_t)PLAT_UC_FP(uc);
    return (uint64_t)PLAT_UC_X(uc, r);
}

static inline void reg_set_xt(mock_ucontext_t *uc, unsigned r, uint64_t val, int is_32bit) {
    if (r == 31) return;
    if (is_32bit) val = (uint32_t)val;
    if (r == 30) PLAT_UC_LR(uc) = val;
    else if (r == 29) PLAT_UC_FP(uc) = val;
    else PLAT_UC_X(uc, r) = val;
}

static _Atomic int g_unaligned_atomic_lock;
static inline void unaligned_atomic_lock(void) {
    while (atomic_exchange_explicit(&g_unaligned_atomic_lock, 1, memory_order_acquire)) {}
}
static inline void unaligned_atomic_unlock(void) {
    atomic_store_explicit(&g_unaligned_atomic_lock, 0, memory_order_release);
}

static int plat_mem_read(uint64_t addr, void *out, size_t len) {
    memcpy(out, (void *)addr, len);
    return 0;
}

static int emulate_unaligned_atomic(void *fault_addr, mock_ucontext_t *uc) {
    uint32_t insn = 0;
    if (plat_mem_read(PLAT_UC_PC(uc), &insn, sizeof insn)) return 0;

    // 1. LSE Atomic memory operations: LDADD, LDCLR, LDEOR, LDSET, LDSMAX, LDSMIN, LDUMAX, LDUMIN, SWP, LDAPR
    if ((insn & 0x3F200C00u) == 0x38200000u) {
        unsigned sz = (insn >> 30) & 3;
        unsigned rs = (insn >> 16) & 0x1F;
        unsigned opc = (insn >> 12) & 0xF;
        unsigned rn = (insn >> 5) & 0x1F;
        unsigned rt = insn & 0x1F;

        if (opc > 8 && opc != 12) return 0;

        uint64_t addr = reg_get_xn(uc, rn);
        uint64_t s = reg_get_xt(uc, rs);
        uint64_t old_val = 0, new_val = 0;

        unaligned_atomic_lock();
        if (sz == 0) {
            uint8_t m; memcpy(&m, (void *)addr, 1); old_val = m;
        } else if (sz == 1) {
            uint16_t m; memcpy(&m, (void *)addr, 2); old_val = m;
        } else if (sz == 2) {
            uint32_t m; memcpy(&m, (void *)addr, 4); old_val = m;
        } else {
            uint64_t m; memcpy(&m, (void *)addr, 8); old_val = m;
        }

        if (opc == 12) {
            new_val = old_val;
        } else {
            switch (opc) {
            case 0:
                if (sz == 0) new_val = (uint8_t)(old_val + s);
                else if (sz == 1) new_val = (uint16_t)(old_val + s);
                else if (sz == 2) new_val = (uint32_t)(old_val + s);
                else new_val = old_val + s;
                break;
            case 1:
                if (sz == 0) new_val = (uint8_t)(old_val & ~(uint8_t)s);
                else if (sz == 1) new_val = (uint16_t)(old_val & ~(uint16_t)s);
                else if (sz == 2) new_val = (uint32_t)(old_val & ~(uint32_t)s);
                else new_val = old_val & ~s;
                break;
            case 2:
                if (sz == 0) new_val = (uint8_t)(old_val ^ (uint8_t)s);
                else if (sz == 1) new_val = (uint16_t)(old_val ^ (uint16_t)s);
                else if (sz == 2) new_val = (uint32_t)(old_val ^ (uint32_t)s);
                else new_val = old_val ^ s;
                break;
            case 3:
                if (sz == 0) new_val = (uint8_t)(old_val | (uint8_t)s);
                else if (sz == 1) new_val = (uint16_t)(old_val | (uint16_t)s);
                else if (sz == 2) new_val = (uint32_t)(old_val | (uint32_t)s);
                else new_val = old_val | s;
                break;
            case 4:
                if (sz == 0) { int8_t a = (int8_t)old_val, b = (int8_t)s; new_val = (uint8_t)(a > b ? a : b); }
                else if (sz == 1) { int16_t a = (int16_t)old_val, b = (int16_t)s; new_val = (uint16_t)(a > b ? a : b); }
                else if (sz == 2) { int32_t a = (int32_t)old_val, b = (int32_t)s; new_val = (uint32_t)(a > b ? a : b); }
                else { int64_t a = (int64_t)old_val, b = (int64_t)s; new_val = (uint64_t)(a > b ? a : b); }
                break;
            case 5:
                if (sz == 0) { int8_t a = (int8_t)old_val, b = (int8_t)s; new_val = (uint8_t)(a < b ? a : b); }
                else if (sz == 1) { int16_t a = (int16_t)old_val, b = (int16_t)s; new_val = (uint16_t)(a < b ? a : b); }
                else if (sz == 2) { int32_t a = (int32_t)old_val, b = (int32_t)s; new_val = (uint32_t)(a < b ? a : b); }
                else { int64_t a = (int64_t)old_val, b = (int64_t)s; new_val = (uint64_t)(a < b ? a : b); }
                break;
            case 6:
                if (sz == 0) { uint8_t a = (uint8_t)old_val, b = (uint8_t)s; new_val = a > b ? a : b; }
                else if (sz == 1) { uint16_t a = (uint16_t)old_val, b = (uint16_t)s; new_val = a > b ? a : b; }
                else if (sz == 2) { uint32_t a = (uint32_t)old_val, b = (uint32_t)s; new_val = a > b ? a : b; }
                else { new_val = old_val > s ? old_val : s; }
                break;
            case 7:
                if (sz == 0) { uint8_t a = (uint8_t)old_val, b = (uint8_t)s; new_val = a < b ? a : b; }
                else if (sz == 1) { uint16_t a = (uint16_t)old_val, b = (uint16_t)s; new_val = a < b ? a : b; }
                else if (sz == 2) { uint32_t a = (uint32_t)old_val, b = (uint32_t)s; new_val = a < b ? a : b; }
                else { new_val = old_val < s ? old_val : s; }
                break;
            case 8:
                if (sz == 0) new_val = (uint8_t)s;
                else if (sz == 1) new_val = (uint16_t)s;
                else if (sz == 2) new_val = (uint32_t)s;
                else new_val = s;
                break;
            }

            if (sz == 0) {
                uint8_t m = (uint8_t)new_val; memcpy((void *)addr, &m, 1);
            } else if (sz == 1) {
                uint16_t m = (uint16_t)new_val; memcpy((void *)addr, &m, 2);
            } else if (sz == 2) {
                uint32_t m = (uint32_t)new_val; memcpy((void *)addr, &m, 4);
            } else {
                uint64_t m = new_val; memcpy((void *)addr, &m, 8);
            }
        }
        unaligned_atomic_unlock();

        reg_set_xt(uc, rt, old_val, sz < 3);
        PLAT_UC_PC(uc) += 4;
        return 1;
    }

    // 2. CAS: CAS, CASA, CASL, CASAL
    if ((insn & 0x3FA07C00u) == 0x08A07C00u) {
        unsigned sz = (insn >> 30) & 3;
        unsigned rs = (insn >> 16) & 0x1F;
        unsigned rn = (insn >> 5) & 0x1F;
        unsigned rt = insn & 0x1F;
        uint64_t addr = reg_get_xn(uc, rn);
        uint64_t s = reg_get_xt(uc, rs);
        uint64_t t = reg_get_xt(uc, rt);
        uint64_t old_val = 0;

        unaligned_atomic_lock();
        if (sz == 0) {
            uint8_t m; memcpy(&m, (void *)addr, 1); old_val = m;
            if ((uint8_t)old_val == (uint8_t)s) { uint8_t nw = (uint8_t)t; memcpy((void *)addr, &nw, 1); }
        } else if (sz == 1) {
            uint16_t m; memcpy(&m, (void *)addr, 2); old_val = m;
            if ((uint16_t)old_val == (uint16_t)s) { uint16_t nw = (uint16_t)t; memcpy((void *)addr, &nw, 2); }
        } else if (sz == 2) {
            uint32_t m; memcpy(&m, (void *)addr, 4); old_val = m;
            if ((uint32_t)old_val == (uint32_t)s) { uint32_t nw = (uint32_t)t; memcpy((void *)addr, &nw, 4); }
        } else {
            uint64_t m; memcpy(&m, (void *)addr, 8); old_val = m;
            if (old_val == s) { uint64_t nw = t; memcpy((void *)addr, &nw, 8); }
        }
        unaligned_atomic_unlock();

        reg_set_xt(uc, rs, old_val, sz < 3);
        PLAT_UC_PC(uc) += 4;
        return 1;
    }

    // 3. Ordered load / store: LDAR, STLR
    if ((insn & 0x3FBF7C00u) == 0x089F7C00u) {
        int is_load = (insn >> 22) & 1;
        unsigned sz = (insn >> 30) & 3;
        unsigned rn = (insn >> 5) & 0x1F;
        unsigned rt = insn & 0x1F;
        uint64_t addr = reg_get_xn(uc, rn);

        unaligned_atomic_lock();
        if (is_load) {
            uint64_t val = 0;
            if (sz == 0) { uint8_t m; memcpy(&m, (void *)addr, 1); val = m; }
            else if (sz == 1) { uint16_t m; memcpy(&m, (void *)addr, 2); val = m; }
            else if (sz == 2) { uint32_t m; memcpy(&m, (void *)addr, 4); val = m; }
            else { uint64_t m; memcpy(&m, (void *)addr, 8); val = m; }
            unaligned_atomic_unlock();
            reg_set_xt(uc, rt, val, sz < 3);
        } else {
            uint64_t val = reg_get_xt(uc, rt);
            if (sz == 0) { uint8_t m = (uint8_t)val; memcpy((void *)addr, &m, 1); }
            else if (sz == 1) { uint16_t m = (uint16_t)val; memcpy((void *)addr, &m, 2); }
            else if (sz == 2) { uint32_t m = (uint32_t)val; memcpy((void *)addr, &m, 4); }
            else { uint64_t m = val; memcpy((void *)addr, &m, 8); }
            unaligned_atomic_unlock();
        }
        PLAT_UC_PC(uc) += 4;
        return 1;
    }

    // 4. Exclusive load: LDAXR, LDXR
    if ((insn & 0x3FFF7C00u) == 0x085F7C00u) {
        unsigned sz = (insn >> 30) & 3;
        unsigned rn = (insn >> 5) & 0x1F;
        unsigned rt = insn & 0x1F;
        uint64_t addr = reg_get_xn(uc, rn);
        uint64_t val = 0;

        unaligned_atomic_lock();
        if (sz == 0) { uint8_t m; memcpy(&m, (void *)addr, 1); val = m; }
        else if (sz == 1) { uint16_t m; memcpy(&m, (void *)addr, 2); val = m; }
        else if (sz == 2) { uint32_t m; memcpy(&m, (void *)addr, 4); val = m; }
        else { uint64_t m; memcpy(&m, (void *)addr, 8); val = m; }
        unaligned_atomic_unlock();

        reg_set_xt(uc, rt, val, sz < 3);
        PLAT_UC_PC(uc) += 4;
        return 1;
    }

    // 5. Exclusive store: STLXR, STXR
    if ((insn & 0x3FE07C00u) == 0x08007C00u) {
        unsigned sz = (insn >> 30) & 3;
        unsigned rs = (insn >> 16) & 0x1F;
        unsigned rn = (insn >> 5) & 0x1F;
        unsigned rt = insn & 0x1F;
        uint64_t addr = reg_get_xn(uc, rn);
        uint64_t val = reg_get_xt(uc, rt);

        unaligned_atomic_lock();
        if (sz == 0) { uint8_t m = (uint8_t)val; memcpy((void *)addr, &m, 1); }
        else if (sz == 1) { uint16_t m = (uint16_t)val; memcpy((void *)addr, &m, 2); }
        else if (sz == 2) { uint32_t m = (uint32_t)val; memcpy((void *)addr, &m, 4); }
        else { uint64_t m = val; memcpy((void *)addr, &m, 8); }
        unaligned_atomic_unlock();

        reg_set_xt(uc, rs, 0, 1);
        PLAT_UC_PC(uc) += 4;
        return 1;
    }

    return 0;
}

int main(void) {
    uint8_t mem[32] = {0};
    uint32_t *target = (uint32_t *)(mem + 1);
    *target = 42;

    mock_ucontext_t uc;
    memset(&uc, 0, sizeof uc);

    // Test 1: LDADD w9, w8, [x8] (0xb8290108)
    uint32_t insn = 0xb8290108;
    uc.pc = (uint64_t)&insn;
    uc.regs[8] = (uint64_t)(mem + 1);
    uc.regs[9] = 1;
    int res = emulate_unaligned_atomic((void *)(mem + 1), &uc);
    assert(res == 1);
    assert(uc.pc == (uint64_t)&insn + 4);
    assert(*target == 43);
    assert((uint32_t)uc.regs[8] == 42);

    // Test 2: LDCLR w9, w8, [x8] (0xb8291108) -> target &= ~1 (43 & ~1 = 42)
    uint32_t insn_clr = 0xb8291108;
    uc.pc = (uint64_t)&insn_clr;
    uc.regs[8] = (uint64_t)(mem + 1);
    uc.regs[9] = 1;
    res = emulate_unaligned_atomic((void *)(mem + 1), &uc);
    assert(res == 1);
    assert(*target == 42);
    assert((uint32_t)uc.regs[8] == 43);

    // Test 3: LDEOR w9, w8, [x8] (0xb8292108) -> target ^= 2 (42 ^ 2 = 40)
    uint32_t insn_eor = 0xb8292108;
    uc.pc = (uint64_t)&insn_eor;
    uc.regs[8] = (uint64_t)(mem + 1);
    uc.regs[9] = 2;
    res = emulate_unaligned_atomic((void *)(mem + 1), &uc);
    assert(res == 1);
    assert(*target == 40);
    assert((uint32_t)uc.regs[8] == 42);

    // Test 4: LDSET w9, w8, [x8] (0xb8293108) -> target |= 3 (40 | 3 = 43)
    uint32_t insn_set = 0xb8293108;
    uc.pc = (uint64_t)&insn_set;
    uc.regs[8] = (uint64_t)(mem + 1);
    uc.regs[9] = 3;
    res = emulate_unaligned_atomic((void *)(mem + 1), &uc);
    assert(res == 1);
    assert(*target == 43);
    assert((uint32_t)uc.regs[8] == 40);

    // Test 5: SWP w9, w8, [x8] (0xb8298108) -> swap target with 100
    uint32_t insn_swp = 0xb8298108;
    uc.pc = (uint64_t)&insn_swp;
    uc.regs[8] = (uint64_t)(mem + 1);
    uc.regs[9] = 100;
    res = emulate_unaligned_atomic((void *)(mem + 1), &uc);
    assert(res == 1);
    assert(*target == 100);
    assert((uint32_t)uc.regs[8] == 43);

    // Test 6: CAS match
    uint32_t cas_insn = (2 << 30) | (0x08 << 24) | (1 << 23) | (1 << 22) | (1 << 21) | (0 << 16) | (1 << 15) | (0x1f << 10) | (1 << 5) | 8;
    uc.pc = (uint64_t)&cas_insn;
    uc.regs[1] = (uint64_t)(mem + 1);
    uc.regs[0] = 100;
    uc.regs[8] = 200;
    res = emulate_unaligned_atomic((void *)(mem + 1), &uc);
    assert(res == 1);
    assert(*target == 200);
    assert(uc.regs[0] == 100);

    // Test 7: CAS mismatch
    uc.pc = (uint64_t)&cas_insn;
    uc.regs[0] = 50;
    uc.regs[8] = 999;
    res = emulate_unaligned_atomic((void *)(mem + 1), &uc);
    assert(res == 1);
    assert(*target == 200);
    assert(uc.regs[0] == 200);

    // Test 8: STLR w9, [x8] (0x889ffd09)
    uint32_t insn_stlr = (2 << 30) | (0x08 << 24) | (1 << 23) | (0 << 22) | (0x1f << 16) | (1 << 15) | (0x1f << 10) | (8 << 5) | 9;
    uc.pc = (uint64_t)&insn_stlr;
    uc.regs[8] = (uint64_t)(mem + 1);
    uc.regs[9] = 777;
    res = emulate_unaligned_atomic((void *)(mem + 1), &uc);
    assert(res == 1);
    assert(*target == 777);

    // Test 9: LDAR w9, [x8] (0x88dffd09)
    uint32_t insn_ldar = (2 << 30) | (0x08 << 24) | (1 << 23) | (1 << 22) | (0x1f << 16) | (1 << 15) | (0x1f << 10) | (8 << 5) | 9;
    uc.pc = (uint64_t)&insn_ldar;
    uc.regs[8] = (uint64_t)(mem + 1);
    uc.regs[9] = 0;
    res = emulate_unaligned_atomic((void *)(mem + 1), &uc);
    assert(res == 1);
    assert((uint32_t)uc.regs[9] == 777);

    // Test 10: STXR w0, w9, [x8] (0x88007d09)
    uint32_t insn_stxr = (2 << 30) | (0x08 << 24) | (0 << 23) | (0 << 22) | (0 << 16) | (0 << 15) | (0x1f << 10) | (8 << 5) | 9;
    uc.pc = (uint64_t)&insn_stxr;
    uc.regs[8] = (uint64_t)(mem + 1);
    uc.regs[9] = 888;
    uc.regs[0] = 99;
    res = emulate_unaligned_atomic((void *)(mem + 1), &uc);
    assert(res == 1);
    assert(*target == 888);
    assert(uc.regs[0] == 0);

    printf("t_android_atomic_emu: ok\n");
    return 0;
}
