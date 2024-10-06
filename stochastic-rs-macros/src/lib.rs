extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, FieldsNamed, Visibility};

#[proc_macro_derive(ImplNew)]
pub fn derive_impl_new(input: TokenStream) -> TokenStream {
  let input = parse_macro_input!(input as DeriveInput);

  let name = input.ident;

  let fields = if let syn::Data::Struct(data) = input.data {
    if let syn::Fields::Named(FieldsNamed { named, .. }) = data.fields {
      named
    } else {
      panic!("`ImplNew` macro can only be used on structs with named fields");
    }
  } else {
    panic!("`ImplNew` macro can only be used on structs");
  };

  let pub_fields = fields
    .iter()
    .filter(|f| matches!(f.vis, Visibility::Public(_)))
    .collect::<Vec<_>>();

  let non_pub_fields: Vec<_> = fields
    .iter()
    .filter(|f| !matches!(f.vis, Visibility::Public(_)))
    .collect::<Vec<_>>();

  let pub_field_names = pub_fields.iter().map(|f| &f.ident).collect::<Vec<_>>();
  let pub_field_types = pub_fields.iter().map(|f| &f.ty).collect::<Vec<_>>();
  let non_pub_field_names = non_pub_fields.iter().map(|f| &f.ident).collect::<Vec<_>>();

  let expanded = quote! {
      impl #name {
        #[must_use]
        pub fn new(#(#pub_field_names: #pub_field_types),*) -> Self {
              Self {
                  #(#pub_field_names),*,
                  #(#non_pub_field_names: Default::default()),*
              }
          }
      }
  };

  TokenStream::from(expanded)
}
