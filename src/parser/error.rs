use std::{error, fmt};

#[derive(Debug, Clone)]
pub enum ParseFieldError {
    /// A segment ID couldn't be parsed as a u64. Can only happen
    /// when parsing into a GFA<u64, T>.
    UintIdError,
    /// A bytestring couldn't be parsed as a bytestring, can happen
    /// when the contents aren't UTF8.
    Utf8Error,
    /// A field couldn't be parsed into the correct type
    ParseFromStringError,
    /// Attempted to parse an orientation that wasn't + or -.
    OrientationError,
    /// A required field was incorrectly formatted. Includes the field
    /// name as defined by the GFA1 spec.
    InvalidField(&'static str),
    MissingFields,
    Unknown,
}

macro_rules! impl_many_from {
    ($to:ty, ($from:ty, $out:expr)) => ();
    ($to:ty, ($from:ty, $out:expr), $(($f:ty, $o:expr)),* $(,)?) => (
        impl From<$from> for $to {
            fn from(_: $from) -> Self {
                $out
            }
        }
        impl_many_from!($to, $(($f, $o)),*);
    );
}

impl_many_from!(
    ParseFieldError,
    (std::str::Utf8Error, ParseFieldError::Utf8Error),
    (
        std::num::ParseIntError,
        ParseFieldError::ParseFromStringError
    ),
    (
        std::num::ParseFloatError,
        ParseFieldError::ParseFromStringError
    )
);

impl fmt::Display for ParseFieldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use ParseFieldError as PFE;
        match self {
            PFE::UintIdError => {
                write!(f, "Failed to parse a segment ID as an unsigned integer")
            }
            PFE::Utf8Error => {
                write!(f, "Failed to parse a bytestring as a UTF-8 string")
            }
            PFE::ParseFromStringError => {
                write!(f, "Failed to parse a field from a string")
            }
            PFE::OrientationError => {
                write!(f, "Failed to parse an orientation character")
            }
            PFE::InvalidField(field) => {
                write!(f, "Failed to parse field `{}`", field)
            }
            PFE::MissingFields => write!(f, "Line is missing required fields"),
            PFE::Unknown => write!(f, "Unknown error when parsing a field"),
        }
    }
}

/// Type encapsulating different kinds of GFA parsing errors
#[derive(Debug)]
pub enum ParseError {
    /// The line type was something other than 'H', 'S', 'L', 'C', or
    /// 'P'. This is ignored by the file parser rather than a fail
    /// condition.
    UnknownLineType,
    /// Tried to parse an empty line. Can be ignored.
    EmptyLine,
    /// A line couldn't be parsed. Includes the problem line and a
    /// variant describing the error.
    InvalidLine(ParseFieldError, String),
    InvalidField(ParseFieldError),
    /// Wrapper for an IO error.
    IOError(std::io::Error),

    Unknown,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use ParseError as PE;
        match self {
            PE::UnknownLineType => {
                write!(f, "Line type was not one of 'H', 'S', 'L', 'C', 'P'")
            }
            PE::EmptyLine => write!(f, "Line was empty"),
            PE::InvalidLine(field_err, line) => {
                write!(f, "Failed to parse line {}, error: {}", line, field_err)
            }
            PE::InvalidField(field_err) => {
                write!(f, "Failed to parse field: {}", field_err)
            }
            PE::IOError(err) => write!(f, "IO error: {}", err),
            PE::Unknown => write!(f, "Unknown error when parsing a line"),
        }
    }
}

impl From<std::io::Error> for ParseError {
    fn from(err: std::io::Error) -> Self {
        Self::IOError(err)
    }
}

impl ParseError {
    pub(crate) fn invalid_line(error: ParseFieldError, line: &[u8]) -> Self {
        let s = std::str::from_utf8(line).unwrap();
        Self::InvalidLine(error, s.into())
    }

    pub fn break_if_necessary(self) -> Result<(), ParseError> {
        if self.can_safely_continue() {
            Ok(())
        } else {
            Err(self)
        }
    }

    pub const fn can_safely_continue(&self) -> bool {
        match self {
            ParseError::EmptyLine => true,
            ParseError::UnknownLineType => true,
            _ => false,
        }
    }
}
