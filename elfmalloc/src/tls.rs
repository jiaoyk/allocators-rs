//! Allocator-safe thread-local storage.
//!
//! The `tls` module implements thread-local storage that, unlike the standard library's
//! implementation, is safe for use in a global allocator.

use std::cell::UnsafeCell;
use std::mem;

/// Declare a thread-local variable.
///
/// `internal_thread_local` (so named in order to avoid a conflict with libstd's `thread_local`)
/// declares a thread-local variable. It is invoked as:
///
/// ```rust,ignore
/// internal_thread_local!{ static <name>: <type> = <expr>; }
/// ```
///
/// For example,
///
/// ```rust,ignore
/// internal_thread_local!{ static FOO: usize = 0; }
/// ```
///
/// Thread-local variables follow a distinct lifecycle, and can be in one of four states:
/// - All thread-local variables start out as *uninitialized*.
/// - When a thread-local variable is first accessed, it is moved into the *initializing* state,
///   and its initializer is called.
/// - Once the initializer returns, the thread-local variable is initialized to the returned value,
///   and it moves into the *initialized* state.
/// - When the thread exits, the variable moves into the *dropped* state, and the variable is
///   dropped.
///
/// Thread-local variables can be accessed using the `with` method. If the variable is in the
/// *uninitialized* or *initialized* states, the variable can be accessed. Otherwise, it cannot,
/// and it is the caller's responsibility to figure out a workaround for its task that does not
/// involve accessing the thread-local variable.
#[macro_export]
macro_rules! internal_thread_local {
    (static $name:ident: $t: ty = $init:expr;) => (
        static $name: $crate::tls::TLSSlot<$t> = {
            fn __init() -> $t { $init }

            fn __drop() { $name.drop(); }

            thread_local!{ static DROPPER: $crate::tls::CallOnDrop = $crate::tls::CallOnDrop(__drop); }

            // DROPPER will only be dropped if it is first initialized, so we provide this function
            // to be called when the TLSSlot is first initialized. The act of calling DROPPER.with
            // will cause DROPPER to be initialized, ensuring that it will later be dropped on
            // thread exit.
            fn __register_dtor() { DROPPER.with(|_| {}); }

            $crate::tls::TLSSlot::new(__init, __register_dtor)
        };
    )
}

#[derive(Eq, PartialEq)]
enum TLSValue<T> {
    Uninitialized,
    Initializing,
    Initialized(T),
    Dropped,
}

/// A slot for a thread-local variable.
///
/// A `TLSSlot` should be initialized using the `internal_thread_local!` macro. See its
/// documentation for details on declaring and using thread-local variables.
pub struct TLSSlot<T> {
    slot: UnsafeCell<TLSValue<T>>,
    init: fn() -> T,
    register_dtor: fn(),
}

impl<T> TLSSlot<T> {
    #[doc(hidden)]
    pub const fn new(init: fn() -> T, register_dtor: fn()) -> TLSSlot<T> {
        TLSSlot {
            slot: UnsafeCell::new(TLSValue::Uninitialized),
            init,
            register_dtor,
        }
    }

    /// Access the TLS slot.
    ///
    /// `with` accepts a function that will be called with a reference to the TLS value. If the
    /// slot is in the *initializing* or *dropped* state, `with` will return `None` without
    /// invoking `f`. If the slot is in the *uninitialized* state, `with` will initialize the value
    /// and then call `f`. If the slot is in the *initialized* state, `with` will call `f`. In
    /// either of these last two cases, `with` will return `Some(r)`, where `r` is the value
    /// returned from the call to `f`.
    pub unsafe fn with<R, F: FnOnce(&mut T) -> R>(&self, f: F) -> Option<R> {
        let ptr = self.slot.get();
        match &mut *ptr {
            &mut TLSValue::Initialized(ref mut t) => return Some(f(t)),
            &mut TLSValue::Uninitialized => {}
            &mut TLSValue::Initializing |
            &mut TLSValue::Dropped => return None,
        }

        // Move into to the Initializing state before registering the destructor in case
        // registering the destructor involves allocation. If it does, the nested access to this
        // TLS value will detect that the value is in state Initializing, the call to with will
        // return None, and a fallback path can be taken.
        *ptr = TLSValue::Initializing;
        (self.register_dtor)();
        *ptr = TLSValue::Initialized((self.init)());
        self.with(f)
    }

    #[doc(hidden)]
    pub fn drop(&self) {
        unsafe {
            let tmp = mem::replace(&mut *self.slot.get(), TLSValue::Dropped);
            mem::drop(tmp);
        }
    }
}

unsafe impl<T> Sync for TLSSlot<T> {}

// The mechanics of registering destructors is complicated and involves a lot of cross-platform
// logic. Instead of implementing that all ourselves, we piggy back on the standard library's
// TLS implementation. Each TLSSlot has a corresponding LocalKey (from the standard library) whose
// value is a CallOnDrop holding a function which will invoke the drop method on the TLSSlot. This
// function is called in CallOnDrop's Drop implementation.
#[doc(hidden)]
pub struct CallOnDrop(fn());

impl Drop for CallOnDrop {
    fn drop(&mut self) {
        (self.0)();
    }
}

#[cfg(test)]
mod tests {
    // Modified from the Rust standard library
    use std::sync::mpsc::{channel, Sender};
    use std::cell::UnsafeCell;
    use std::thread;

    struct Foo(Sender<()>);

    impl Drop for Foo {
        fn drop(&mut self) {
            let Foo(ref s) = *self;
            s.send(()).unwrap();
        }
    }

    #[test]
    fn smoke_dtor() {
        internal_thread_local!{ static FOO: UnsafeCell<Option<Foo>> = UnsafeCell::new(None); }

        let (tx, rx) = channel();
        let _t = thread::spawn(move || unsafe {
                                   let mut tx = Some(tx);
                                   FOO.with(|f| { *f.get() = Some(Foo(tx.take().unwrap())); });
                               });
        rx.recv().unwrap();
    }
}
